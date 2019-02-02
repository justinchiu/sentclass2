import math

from .base import Sent
from .crfneg import CrfNeg
from .. import sentihood as data

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from pyro.ops.contract import ubersum

class CrfNegT(CrfNeg):
    def __init__(
        self,
        V = None,
        L = None,
        A = None,
        S = None,
        emb_sz = 256,
        rnn_sz = 256,
        nlayers = 2,
        dp = 0.3,
    ):
        super(CrfNegT, self).__init__(V, L, A, S, emb_sz, rnn_sz, nlayers, dp)

        num_loc = len(L) if L is not None else 1
        num_asp = len(A) if A is not None else 1

        # Score each sentiment for each location and aspect
        # Store the combined pos, neg, none in a single vector :(
        if self.outer_plate:
            self.proj_s = nn.Parameter(torch.randn(len(L)*len(A), len(S), emb_sz))
            self.proj_s.data[:,:,self.S.stoi["none"]].mul_(2)
            self.proj_neg = nn.Parameter(torch.randn(len(L)*len(A), 2, 2*rnn_sz))
        else:
            self.proj_s = nn.Linear(emb_sz, len(S))
            self.proj_neg = nn.Linear(2*rnn_sz, 2)
        self.psi_ys = nn.Parameter(torch.FloatTensor([0.1, 0.1, 0.1]))
        #self.phi_b = nn.Parameter(torch.FloatTensor([0.8, 0.2]))
        self.phi_b = nn.Parameter(torch.FloatTensor([1, 1]))
        #self.phi_b.requires_grad = False
        self.flip = nn.Parameter(
            torch.Tensor([
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ])
        )
        self.flip.requires_grad = False
        self.psi_bb = nn.Parameter(torch.randn(2,2))

    def forward(self, x, lens, a, l):
        words = x

        emb = self.drop(self.lut(x))
        p_emb = pack(emb, lens, True)

        N = x.shape[0]
        T = x.shape[1]

        state = None
        if self.outer_plate:
            y_idx = l * len(self.A) + a if self.L is not None else a
            s = (self.lut_la(y_idx)
                .view(N, 2, 2 * self.nlayers, self.rnn_sz)
                .permute(1, 2, 0, 3)
                .contiguous())
            state = (s[0], s[1])
        x, (h, c) = self.rnn(p_emb, state)
        # h: L * D x N x H
        x = unpack(x, True)[0]
        #import pdb; pdb.set_trace()

        phi_s, phi_neg = None, None
        if self.outer_plate:
            proj_s = self.proj_s[y_idx.squeeze(-1)]
            phi_s = torch.einsum("nsh,nth->nts", [proj_s, emb])
            proj_neg = self.proj_neg[y_idx.squeeze(-1)]
            phi_neg = torch.einsum("nbh,nth->ntb", [proj_neg, x])
        else:
            phi_s = self.proj_s(emb)
            phi_neg = self.proj_neg(x)
        # CONV
        if self.outer_plate:
            c = (self.conv(emb.transpose(-1, -2))
                .transpose(-1, -2)
                .view(N,T,-1,2*self.rnn_sz))#[:,:,y_idx.squeeze(-1),:]
            cy = c.gather(2, y_idx.view(N,1,1,1).expand(N,T,1,100)).squeeze(-2)
        #phi_neg = torch.einsum("nbh,nth->ntb", [proj_neg, cy])
        # /CONV
        # add prior
        phi_neg = phi_neg + self.phi_b.view(1, 1, 2)

        psi_ybs0 = torch.diag(self.psi_ys)
        psi_ybs1 = psi_ybs0 @ self.flip
        psi_ybs = (torch.stack([psi_ybs0, psi_ybs1], 1)
        #psi_ybs = (torch.stack([psi_ybs0 @ self.fm1, psi_ybs0 @ self.fm2], 1)
            .view(1, 1, len(self.S), 2, len(self.S))
            .repeat(N, T, 1, 1, 1))

        phi_b = self.phi_b.view(1, 1, 2).repeat(N, T, 1)
        psi_bb = self.psi_bb.view(1, 1, 2, 2).repeat(N, T, 1, 1)
        # phi_s | x: N x T x S
        # phi_neg | x: N x T x 2
        # phi_b: N x T x 2
        # psi_bb: N x T x 2 x 2
        # psi_ybs: N x T x Y x 2 x S
         
        idxs = torch.arange(0, max(lens)).to(lens.device)
        # mask: N x R
        mask = (idxs.repeat(len(lens), 1) >= lens.unsqueeze(-1))
        phi_s.masked_fill_(mask.unsqueeze(-1), 0)
        phi_neg.masked_fill_(mask.unsqueeze(-1), 0)
        phi_b.masked_fill_(mask.unsqueeze(-1), 0)
        psi_bb.masked_fill_(mask.view(N, T, 1, 1), 0)
        #psi_bb[:,0].fill_(0)
        
        #print(phi_neg)
        #import pdb; pdb.set_trace()
        if self.training:
            self._N += 1
        #if self._N > 10 and self.training:
        if self.training and T < 5:
            # marginal density b
            phi_yl, phi_sl, phi_negl, phi_bl, psi_bbl, psi_ybsl = self.potentials(
                phi_s, phi_neg, phi_b, psi_bb, psi_ybs)
            hs = torch.stack(
                self.marginal_s(phi_yl, phi_sl, phi_negl, phi_bl, psi_bbl, psi_ybsl),
                1
            )
            ps = hs.softmax(-1)

            phi_yl, phi_sl, phi_negl, phi_bl, psi_bbl, psi_ybsl = self.potentials(
                phi_s, phi_neg, phi_b, psi_bb, psi_ybs)
            hb = torch.stack(
                self.marginal_b(phi_yl, phi_sl, phi_negl, phi_bl, psi_bbl, psi_ybsl),
                1
            )
            pb = hb.softmax(-1)
            import pdb; pdb.set_trace()
            def stuff(i):
                #loc = self.L.itos[l[i]]
                asp = self.A and self.A.itos[a[i]]
                return self.tostr(words[i]), None, asp, xp[i], yp[i], bp[i]
                import pdb; pdb.set_trace()
            # wordsi, loc, asp, xpi, ypi, bpi = stuff(10)       psi_ybs.masked_fill_(mask.view(N, T, 1, 1, 1).expand_as(psi_ybs), 0)

        # SPLIT...MAINTAINS REFERENCE TO ORIGINAL VALUES
        phi_y = torch.zeros(N, len(self.S)).to(self.lut.weight.device)
        phi_sl   = [x.squeeze(1) for x in phi_s.split(1, 1)]
        phi_negl = [x.squeeze(1) for x in phi_neg.split(1, 1)]
        phi_bl   = [x.squeeze(1) for x in phi_b.split(1, 1)]
        psi_bbl  = [x.squeeze(1) for x in psi_bb.split(1, 1)]
        psi_ybsl = [x.squeeze(1) for x in psi_ybs.split(1, 1)]
        psi_bbl[0] = psi_bbl[0][:,0].fill_(0)
        """
        # phi_y : N x Y
        # phi_s | x: N x T x S
        # phi_neg | x: N x T x 2
        # phi_b: N x T x 2
        # psi_bb: N x T x 2 x 2
        # psi_ybs: N x T x Y x 2 x S
        """
        for t in range(T):
            # marginalize over b_t and s_t
            b = phi_negl[t] + phi_bl[t] + psi_bbl[t]
            # update next psi_bb
            if t < T-1:
                psi_bbl[t+1] = torch.logsumexp(psi_bbl[t+1] + b.unsqueeze(-1), dim=-2)
            phi_y = phi_y + torch.logsumexp(torch.logsumexp(
                psi_ybsl[t] + b.view(N, 1, 2, 1) + phi_sl[t].view(N, 1, 1, len(self.S)),
            dim=-1), dim=-1)
        #import pdb; pdb.set_trace()
        return phi_y

    def potentials(self, phi_s, phi_neg, phi_b, psi_bb, psi_ybs):
        N = psi_ybs.shape[0]
        phi_y = torch.zeros(N, len(self.S), 1).to(self.lut.weight.device)
        phi_sl   = [x.squeeze(1) for x in phi_s.clone().split(1, 1)]
        phi_negl = [x.squeeze(1) for x in phi_neg.clone().split(1, 1)]
        phi_bl   = [x.squeeze(1) for x in phi_b.clone().split(1, 1)]
        psi_bbl  = [x.squeeze(1) for x in psi_bb.clone().split(1, 1)]
        psi_ybsl = [x.squeeze(1) for x in psi_ybs.clone().split(1, 1)]
        psi_bbl[0] = psi_bbl[0][:,0].fill_(0)
        return phi_y, phi_sl, phi_negl, phi_bl, psi_bbl, psi_ybsl

    def marginal_s(self, phi_ys, phi_sl, phi_negl, phi_bl, psi_bbl, psi_ybsl):
        T = len(psi_ybsl)
        N, Y, B, S = psi_ybsl[0].shape
        py = torch.zeros(N, S).to(phi_ys)
        # compute marginalized s factors
        myb = [(psi_ybsl[t] + phi_sl[t].view(N,1,1,len(self.S))).logsumexp(-1) for t in range(T)]
        # fwd-bwd
        alphas = [0 for x in myb]
        psi_ybbl_a = [x.clone().unsqueeze(1) for x in psi_bbl]
        for t in range(T):
            b = phi_negl[t] + phi_bl[t] 
            yb = myb[t] + b.unsqueeze(-2) + (psi_ybbl_a[t] if t > 0 else 0)
            if t < T-1:
                psi_ybbl_a[t+1] = (psi_ybbl_a[t+1] + yb.unsqueeze(-1)).logsumexp(-2)
            # inclusion
            ybs = (psi_ybsl[t]
                + phi_sl[t].view(N,1,1,len(self.S))
                + b.view(N,1,2,1)
                + (psi_ybbl_a[t].view(N,-1,2,1) if t > 0 else 0)
            )
            alphas[t] = ybs
        # bwd betas
        betas = [0 for x in myb]
        psi_ybbl_b = [x.clone().unsqueeze(1) for x in psi_bbl]
        for t in range(T-1, -1, -1):
            b = phi_negl[t] + phi_bl[t]
            yb = myb[t] + b.unsqueeze(-2) + (psi_ybbl_b[t+1] if t < T-1 else 0)
            if t > 0:
                psi_ybbl_b[t] = (psi_ybbl_b[t] + yb.unsqueeze(-2)).logsumexp(-1)
            # exclusion
            betas[t] = (psi_ybbl_b[t+1] if t < T-1 else psi_ybbl_b[0]).unsqueeze(-1)
        ybsl = [a+b for a, b in zip(alphas, betas)]
        sl = [ybs.logsumexp(-2).logsumexp(-2) for ybs in ybsl]
        import pdb; pdb.set_trace()
        if T == 3:
            # check marginal s
            print("marg s")
            Z,A,B,C = ubersum(
                "na,nab,nb,nbc,nc,nyaA,nybB,nycC,nA,nB,nC->n,nA,nB,nC",
                phi_negl[0]+phi_bl[0], psi_bbl[1],
                phi_negl[1]+phi_bl[1], psi_bbl[2],
                phi_negl[2]+phi_bl[2],
                psi_ybsl[0], psi_ybsl[1], psi_ybsl[2],
                phi_sl[0], phi_sl[1], phi_sl[2],
                modulo_total=True)
            print(Z)
            print(phi_sl[-1].logsumexp(-1))
            print("fun")
            import opt_einsum
            A0 = opt_einsum.contract_path(
                "na,nab,nb,nbc,nc,nyaA,nybB,nycC,nA,nB,nC->nB",
                phi_negl[0]+phi_bl[0], psi_bbl[1],
                phi_negl[1]+phi_bl[1], psi_bbl[2],
                phi_negl[2]+phi_bl[2],
                psi_ybsl[0], psi_ybsl[1], psi_ybsl[2],
                phi_sl[0], phi_sl[1], phi_sl[2])
            print("nB", A0[1])
            A0 = opt_einsum.contract_path(
                "na,nab,nb,nbc,nc,nyaA,nybB,nycC,nA,nB,nC->nC",
                phi_negl[0]+phi_bl[0], psi_bbl[1],
                phi_negl[1]+phi_bl[1], psi_bbl[2],
                phi_negl[2]+phi_bl[2],
                psi_ybsl[0], psi_ybsl[1], psi_ybsl[2],
                phi_sl[0], phi_sl[1], phi_sl[2])
            print("nC", A0[1])
            print((C-Z.unsqueeze(-1)).exp())
            print((sl[-1] - sl[-1].logsumexp(-1, True)).exp())
            import pdb; pdb.set_trace()
        """
        # OLD
        alphas = [x for x in phi_bl]
        psi_bbl_a = [x.clone() for x in psi_bbl]
        for t in range(T):
            b = phi_negl[t] + phi_bl[t] + (psi_bbl_a[t] if t > 0 else 0)
            if t < T-1:
                psi_bbl_a[t+1] = (psi_bbl_a[t+1] + b.unsqueeze(-1)).logsumexp(-2)
            # inclusion
            alphas[t] = b
        # bwd betas
        betas = [x for x in phi_bl]
        psi_bbl_b = [x.clone() for x in psi_bbl]
        for t in range(T-1, -1, -1):
            b = phi_negl[t] + phi_bl[t] + (psi_bbl_b[t+1] if t < T-1 else 0)
            if t > 0:
                psi_bbl_b[t] = (psi_bbl_b[t] + b.unsqueeze(-2)).logsumexp(-1)
            # exclusion
            betas[t] = psi_bbl_b[t+1] if t < T-1 else psi_bbl_b[0]
        bl = [a+b for a, b in zip(alphas, betas)]

        # for each s_t marginalize over all other s_i
        psi_ybsl = [psi_ybsl[t] + phi_sl[t].view(N, 1, 1, len(self.S)) for t in range(T)]
        # needs to be in the same order as above...? oops still a mistake here
        # need to do forward backward on b to the proper messages...
        for t in range(T):
            ybsl = [x for x in psi_ybsl]
            y_t = sum([
                (ybsl[i] + (
                    alphas[i] if i < t else betas[i]
                ).view(N, 1, 2, 1)).logsumexp(-1).logsumexp(-1)
                for i in range(T) if i != t
            ])
            phi_sl[t] = (
                psi_ybsl[t]
                + y_t.view(N, len(self.S), 1, 1)
                + alphas[t].view(N, 1, 2, 1)
                + betas[t].view(N, 1, 2, 1)
            ).logsumexp(-2).logsumexp(-2)
        """
        import pdb; pdb.set_trace()
        return phi_sl

    def marginal_b(self, phi_yb, phi_sl, phi_negl, phi_bl, psi_bbl, psi_ybsl):
        T = len(psi_ybsl)
        N, Y, B, S = psi_ybsl[0].shape
        # marginalize over s's
        for t in range(T):
            phi_yb = phi_yb + (psi_ybsl[t] + phi_sl[t].view(N, 1, 1, len(self.S))).logsumexp(-1)
        phi_b = phi_yb.logsumexp(-2)
        phi_bl = [b + phi_b for b in phi_bl]
        # get marginal for b's
        # fwd alphas
        alphas = [x for x in phi_bl]
        psi_bbl_a = [x.clone() for x in psi_bbl]
        for t in range(T):
            b = phi_negl[t] + phi_bl[t] + psi_bbl_a[t]
            if t < T-1:
                psi_bbl_a[t+1] = (psi_bbl_a[t+1] + b.unsqueeze(-1)).logsumexp(-2)
            alphas[t] = b
        # bwd betas
        betas = [x for x in phi_bl]
        psi_bbl_b = [x.clone() for x in psi_bbl]
        psi_bbl_b.append(psi_bbl_b[0])
        for t in range(T-1, -1, -1):
            b = phi_negl[t] + phi_bl[t] + (psi_bbl_b[t+1] if t < T-1 else 0)
            if t > 0:
                psi_bbl_b[t] = (psi_bbl_b[t] + b.unsqueeze(-2)).logsumexp(-1)
            betas[t] = b
        b = [a+b for a, b in zip(alphas, betas)]
        import pdb; pdb.set_trace()
        return b

    def observe(self, x, lens, l, a, y):
        raise NotImplementedError
