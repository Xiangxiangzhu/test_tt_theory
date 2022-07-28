from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from sublayer import MultiHeadAttention, PositionwiseFeedForward
from sublayer import MN, PN
from sublayer import UniformHeadAttention


# noinspection PyAbstractClass
class GATLayer(nn.Module):
    def __init__(self, output_dim, nheads, dropout=0):
        super(GATLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(nheads, output_dim, output_dim // 4, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(output_dim, output_dim, dropout=dropout)

    def forward(self, x, adj):
        x = self.slf_attn(x, adj)
        x = self.pos_ffn(x)
        return x


# noinspection PyAbstractClass
class ATLayer(nn.Module):
    def __init__(self, output_dim, nheads, dropout=0):
        super(ATLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(nheads, output_dim, output_dim // 4, dropout=dropout)
        # self.pos_ffn = PositionwiseFeedForward(output_dim, output_dim, dropout=dropout)

    def forward(self, x, adj):
        x = self.slf_attn(x, adj)
        # x = self.pos_ffn(x)
        return x


# noinspection PyAbstractClass
class GOTLayer(nn.Module):
    def __init__(self, output_dim, nheads, dropout=0):
        super(GOTLayer, self).__init__()
        self.pos_ffn = PositionwiseFeedForward(output_dim, output_dim, dropout=dropout)

    def forward(self, x, adj):
        x = self.pos_ffn(x)
        return x


# noinspection PyAbstractClass
class NGTLayer(nn.Module):
    def __init__(self, output_dim, nheads, dropout=0):
        super(NGTLayer, self).__init__()
        self.slf_attn = MN(nheads, output_dim, output_dim // 4, dropout=dropout)
        self.pos_ffn = PN(output_dim, output_dim, dropout=dropout)

    def forward(self, x, adj):
        x = self.slf_attn(x, adj)
        x = self.pos_ffn(x)
        return x


# noinspection PyAbstractClass
class UNiTLayer(nn.Module):
    def __init__(self, output_dim, nheads, dropout=0):
        super(UNiTLayer, self).__init__()
        self.slf_attn = UniformHeadAttention(nheads, output_dim, output_dim // 4, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(output_dim, output_dim, dropout=dropout)

    def forward(self, x, adj):
        x = self.slf_attn(x, adj)
        x = self.pos_ffn(x)
        return x


###########################################################################################

class GATNNA(nn.Module):
    def __init__(self, input_dim, output_dim, mask_dim, emb_dim, window_size, nheads, adj_base):
        super(GATNNA, self).__init__()
        self.upp = nn.Linear(window_size, emb_dim)
        self.gat1 = GATLayer(emb_dim, nheads, 0)
        self.down = nn.Linear(mask_dim * emb_dim, output_dim)
        self.adj = adj_base

    def forward(self, x, adj=None):
        if adj is None:
            adj = torch.cat([self.adj for _ in range(x.shape[0])], dim=0).to(torch.float32)
        x = self.upp(x)
        x = self.gat1(x, adj.to(x.device))
        x = x.view(x.shape[0], -1)
        x = self.down(x)
        return x


class GATNNQ(nn.Module):
    def __init__(self, input_state_dim, input_action_dim, output_dim, mask_dim, emb_dim, window_size, nheads, adj_base):
        super(GATNNQ, self).__init__()
        self.upp = nn.Linear(window_size, emb_dim)
        self.gat1 = GATLayer(emb_dim, nheads, 0)
        self.down = nn.Linear(emb_dim, 1)
        self.out1 = nn.Linear(input_state_dim + input_action_dim, emb_dim)
        self.out2 = nn.Linear(emb_dim, output_dim)
        self.adj = adj_base

    def forward(self, x_state, x_action, adj=None):
        if adj is None:
            adj = torch.cat([self.adj for _ in range(x_state.shape[0])], dim=0).to(torch.float32)
        x = self.upp(x_state)
        x = self.gat1(x, adj)
        x = self.down(x)
        x = x.squeeze(-1)
        x = torch.cat([x, x_action], dim=1)
        x = self.out2(F.leaky_relu(self.out1(x)))
        return x


######################

class GOTNNA(nn.Module):
    def __init__(self, input_dim, output_dim, mask_dim, emb_dim, window_size, nheads, adj_base):
        super(GOTNNA, self).__init__()
        self.upp = nn.Linear(window_size, emb_dim)
        self.gat1 = GOTLayer(emb_dim, nheads, 0)
        self.down = nn.Linear(mask_dim * emb_dim, output_dim)
        self.adj = adj_base

    def forward(self, x, adj=None):
        if adj is None:
            adj = torch.cat([self.adj for _ in range(x.shape[0])], dim=0).to(torch.float32)
        x = self.upp(x)
        x = self.gat1(x, adj)
        x = x.view(x.shape[0], -1)
        x = self.down(x)
        return x


class GOTNNQ(nn.Module):
    def __init__(self, input_state_dim, input_action_dim, output_dim, mask_dim, emb_dim, window_size, nheads, adj_base):
        super(GOTNNQ, self).__init__()
        self.upp = nn.Linear(window_size, emb_dim)
        self.gat1 = GOTLayer(emb_dim, nheads, 0)
        self.down = nn.Linear(emb_dim, 1)
        self.out1 = nn.Linear(input_state_dim + input_action_dim, emb_dim)
        self.out2 = nn.Linear(emb_dim, output_dim)
        self.adj = adj_base

    def forward(self, x_state, x_action, adj=None):
        if adj is None:
            adj = torch.cat([self.adj for _ in range(x_state.shape[0])], dim=0).to(torch.float32)
        x = self.upp(x_state)
        x = self.gat1(x, adj)
        x = self.down(x)
        x = x.squeeze(-1)
        x = torch.cat([x, x_action], dim=1)
        x = self.out2(F.leaky_relu(self.out1(x)))
        return x


######################

class ATNNA(nn.Module):
    def __init__(self, input_dim, output_dim, mask_dim, emb_dim, window_size, nheads, adj_base):
        super(ATNNA, self).__init__()
        self.upp = nn.Linear(window_size, emb_dim)
        self.gat1 = ATLayer(emb_dim, nheads, 0)
        self.down = nn.Linear(mask_dim * emb_dim, output_dim)
        self.adj = adj_base

    def forward(self, x, adj=None):
        if adj is None:
            adj = torch.cat([self.adj for _ in range(x.shape[0])], dim=0).to(torch.float32)
        x = self.upp(x)
        x = self.gat1(x, adj)
        x = x.view(x.shape[0], -1)
        x = self.down(x)
        return x


class ATNNQ(nn.Module):
    def __init__(self, input_state_dim, input_action_dim, output_dim, mask_dim, emb_dim, window_size, nheads, adj_base):
        super(ATNNQ, self).__init__()
        self.upp = nn.Linear(window_size, emb_dim)
        self.gat1 = ATLayer(emb_dim, nheads, 0)
        self.down = nn.Linear(emb_dim, 1)
        self.out1 = nn.Linear(input_state_dim + input_action_dim, emb_dim)
        self.out2 = nn.Linear(emb_dim, output_dim)
        self.adj = adj_base

    def forward(self, x_state, x_action, adj=None):
        if adj is None:
            adj = torch.cat([self.adj for _ in range(x_state.shape[0])], dim=0).to(torch.float32)
        x = self.upp(x_state)
        x = self.gat1(x, adj)
        x = self.down(x)
        x = x.squeeze(-1)
        x = torch.cat([x, x_action], dim=1)
        x = self.out2(F.leaky_relu(self.out1(x)))
        return x


######################

class TNNA(nn.Module):
    def __init__(self, input_dim, output_dim, mask_dim, emb_dim, window_size, nheads, adj_base):
        super(TNNA, self).__init__()
        self.upp = nn.Linear(window_size, emb_dim)
        self.gat1 = NGTLayer(emb_dim, nheads, 0)
        self.down = nn.Linear(mask_dim * emb_dim, output_dim)
        self.adj = adj_base

    def forward(self, x, adj=None):
        if adj is None:
            adj = torch.cat([self.adj for _ in range(x.shape[0])], dim=0).to(torch.float32)
        x = self.upp(x)
        x = self.gat1(x, adj)
        x = x.view(x.shape[0], -1)
        x = self.down(x)
        return x


class TNNQ(nn.Module):
    def __init__(self, input_state_dim, input_action_dim, output_dim, mask_dim, emb_dim, window_size, nheads, adj_base):
        super(TNNQ, self).__init__()
        self.upp = nn.Linear(window_size, emb_dim)
        self.gat1 = NGTLayer(emb_dim, nheads, 0)
        self.down = nn.Linear(emb_dim, 1)
        self.out1 = nn.Linear(input_state_dim + input_action_dim, emb_dim)
        self.out2 = nn.Linear(emb_dim, output_dim)
        self.adj = adj_base

    def forward(self, x_state, x_action, adj=None):
        if adj is None:
            adj = torch.cat([self.adj for _ in range(x_state.shape[0])], dim=0).to(torch.float32)
        x = self.upp(x_state)
        x = self.gat1(x, adj)
        x = self.down(x)
        x = x.squeeze(-1)
        x = torch.cat([x, x_action], dim=1)
        x = self.out2(F.leaky_relu(self.out1(x)))
        return x


######################
class UNTNNA(nn.Module):
    def __init__(self, input_dim, output_dim, mask_dim, emb_dim, window_size, nheads, adj_base):
        super(UNTNNA, self).__init__()
        self.upp = nn.Linear(window_size, emb_dim)
        self.gat1 = UNiTLayer(emb_dim, nheads, 0)
        self.down = nn.Linear(mask_dim * emb_dim, output_dim)
        self.adj = adj_base

    def forward(self, x, adj=None):
        if adj is None:
            adj = torch.cat([self.adj for _ in range(x.shape[0])], dim=0).to(torch.float32)
        x = self.upp(x)
        x = self.gat1(x, adj)
        x = x.view(x.shape[0], -1)
        x = self.down(x)
        return x


class UNTNNQ(nn.Module):
    def __init__(self, input_state_dim, input_action_dim, output_dim, mask_dim, emb_dim, window_size, nheads, adj_base):
        super(UNTNNQ, self).__init__()
        self.upp = nn.Linear(window_size, emb_dim)
        self.gat1 = UNiTLayer(emb_dim, nheads, 0)
        self.down = nn.Linear(emb_dim, 1)
        self.out1 = nn.Linear(input_state_dim + input_action_dim, emb_dim)
        self.out2 = nn.Linear(emb_dim, output_dim)
        self.adj = adj_base

    def forward(self, x_state, x_action, adj=None):
        if adj is None:
            adj = torch.cat([self.adj for _ in range(x_state.shape[0])], dim=0).to(torch.float32)
        x = self.upp(x_state)
        x = self.gat1(x, adj)
        x = self.down(x)
        x = x.squeeze(-1)
        x = torch.cat([x, x_action], dim=1)
        x = self.out2(F.leaky_relu(self.out1(x)))
        return x

######################
