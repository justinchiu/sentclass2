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

        phi_y = torch.zeros(N, len(self.S)).to(self.lut.weight.device)
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
        psi_ybs.masked_fill_(mask.view(N, T, 1, 1, 1).expand_as(psi_ybs), 0)

        # is indexing faster (tranpose first?) or splitting?
        # SPLIT?
        #"""
        phi_sl   = [x.squeeze(1) for x in phi_s.split(1, 1)]
        phi_negl = [x.squeeze(1) for x in phi_neg.split(1, 1)]
        phi_bl   = [x.squeeze(1) for x in phi_b.split(1, 1)]
        psi_bbl  = [x.squeeze(1) for x in psi_bb.split(1, 1)]
        psi_ybsl = [x.squeeze(1) for x in psi_ybs.split(1, 1)]
        #"""
        """
        #[for s, n, b, bb, ybs in zip(phi_s, phi_neg, phi_b, psi_bb, psi_ybs) for y in x]
        # Ns, Nn, Nb, Nba, Nybs 
        ok = [y for x in zip(phi_s, phi_neg, phi_b, psi_bb, psi_ybs) for y in x]
        import pdb; pdb.set_trace()

        expr = ""
        args = []
        for t in range(T):
            # phi_s | x: N x T x S
            # phi_neg | x: N x T x 2
            # phi_b: N x T x 2
            # psi_bb: N x T x 2 x 2
            # psi_ybs: N x T x Y x 2 x S
            pass

        """
        """
        # transpose and marginalize by hand; this doesn't work since i want to mutate
        phi_s   = phi_s.transpose(0, 1)
        phi_neg = phi_neg.transpose(0, 1)
        phi_b   = phi_b.transpose(0, 1)
        psi_bb  = psi_bb.transpose(0, 1)
        psi_ybs = psi_ybs.transpose(0, 1)
        """
        psi_bbl[0] = psi_bbl[0][:,0].fill_(0)
        for t in range(T):
            #b = torch.logsumexp(phi_b[t].unsqueeze(-2) + phi_bb[t], dim=-2)
            # marginalize over b_t and s_t
            b = phi_negl[t] + phi_bl[t] + psi_bbl[t]
            # update next psi_bb
            if t < T-1:
                psi_bbl[t+1] = torch.logsumexp(psi_bbl[t+1] + b.unsqueeze(-1), dim=-2)
            phi_y = phi_y + torch.logsumexp(torch.logsumexp(
                psi_ybsl[t] + b.view(N, 1, 2, 1) + phi_sl[t].view(N, 1, 1, len(self.S)),
            dim=-1), dim=-1)
        #print(phi_neg)
        #import pdb; pdb.set_trace()
        if self.training:
            self._N += 1
        if self._N > 50 and self.training:
            # marginal densities?

            """
            Zt, hx, hb = ubersum(
                "nts,ntb,ntybs,ny->nt,nts,ntb",
                phi_s, phi_neg, psi_ybs, phi_y, batch_dims="t", modulo_total=True)
            xp = (hx - Zt.unsqueeze(-1)).exp()
            bp = (hb - Zt.unsqueeze(-1)).exp()
            yp = (hy - Z.unsqueeze(-1)).exp()
            """
            def stuff(i):
                #loc = self.L.itos[l[i]]
                asp = self.A and self.A.itos[a[i]]
                return self.tostr(words[i]), None, asp, xp[i], yp[i], bp[i]
                import pdb; pdb.set_trace()
            # wordsi, loc, asp, xpi, ypi, bpi = stuff(10)
        return phi_y


    def observe(self, x, lens, l, a, y):
        raise NotImplementedError