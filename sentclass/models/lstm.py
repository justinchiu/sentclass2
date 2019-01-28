from .base import Sent
from .. import sentihood as data

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class Lstm(Sent):
    def __init__(
        self,
        V = None,
        L = None,
        A = None,
        S = None,
        final = False,
        emb_sz = 256,
        rnn_sz = 256,
        nlayers = 2,
        dp = 0.3,
    ):
        super(Lstm, self).__init__(V, L, A, S)

        num_loc = len(L) if L is not None else 1
        num_asp = len(A) if A is not None else 1

        self.emb_sz = emb_sz
        self.rnn_sz = rnn_sz
        self.nlayers = nlayers
        self.dp = dp
        self.final = final

        self.lut = nn.Embedding(
            num_embeddings = len(V),
            embedding_dim = emb_sz,
            padding_idx = V.stoi[self.PAD],
        )
        self.lut.weight.data.copy_(V.vectors)
        self.lut.weight.requires_grad = False
        self.lut.weight.data[2].copy_(torch.randn(emb_sz))
        self.lut.weight.data[3].copy_(torch.randn(emb_sz))
        if self.outer_plate:
            self.lut_la = nn.Embedding(
                num_embeddings = num_loc * num_asp,
                embedding_dim = nlayers * 2 * 2 * rnn_sz,
            )
        self.rnn = nn.LSTM(
            input_size = emb_sz,
            hidden_size = rnn_sz,
            num_layers = nlayers,
            bias = True,
            dropout = dp,
            bidirectional = True,
            batch_first   = True,
        )
        self.drop = nn.Dropout(dp)

        # Score each sentiment for each location and aspect
        # Store the combined pos, neg, none in a single vector :(
        self.proj = nn.Linear(
            in_features = 2 * rnn_sz,
            out_features = num_loc * num_asp * len(S),
            bias = False,
        )
        
        #import torch.nn.init
        #for p in self.parameters():
        #    if p.requires_grad and p.dim() == 2:
        #        torch.nn.init.xavier_uniform_(p)


    def forward(self, x, lens, a, l):
        # model takes as input the text, aspect, and location
        # runs BLSTM over text using embedding(location, aspect) as
        # the initial hidden state, as opposed to a different lstm for every pair???
        # output sentiment
        emb = self.drop(self.lut(x))
        p_emb = pack(emb, lens, True)

        N, T = x.shape

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
        # Get the last hidden states for both directions, POSSIBLE BUGS
        if self.final:
            h = (h
                .view(self.nlayers, 2, -1, self.rnn_sz)[-1]
                .permute(1, 0, 2)
                .contiguous()
                .view(-1, 2 * self.rnn_sz))
            h = self.drop(h)
            return self.proj(h)
        else:
            return self.proj(unpack(x, True)[0].max(1)[0])


    def observe(self, x, lens, l, a, y):
        emb = self.drop(self.lut(x))
        p_emb = pack(emb, lens, True)

        N = a.shape[0]

        state = None
        if self.init_state:
            y_idx = l * len(self.A) + a if self.L is not None else a
            s = (self.lut_la(y_idx)
                .view(N, 2, 2 * self.nlayers, self.rnn_sz)
                .permute(1, 2, 0, 3)
                .contiguous())
            state = (s[0], s[1])
        x, (h, c) = self.rnn(p_emb, state)
        ok = self.proj(unpack(x, True)[0]).view(N, -1, len(self.A), len(self.S))
        return ok[:,:,0,:]

