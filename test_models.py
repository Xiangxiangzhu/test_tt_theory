from nn_builder.pytorch.NN import NN
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


def create_NN(input_dim, output_dim, hyperparameters=None):
    """Creates a neural network for the agents to use"""
    if hyperparameters is None:
        hyperparameters = {
            "learning_rate": 1e-4,
            "linear_hidden_units": [64, 32],
            "final_layer_activation": None,
            "batch_norm": False,
            "initialiser": "Xavier"
        }

    default_hyperparameter_choices = {"output_activation": None, "hidden_activations": "relu",
                                      "dropout": 0.0,
                                      "initialiser": "default", "batch_norm": False,
                                      "columns_of_data_to_be_embedded": [],
                                      "embedding_dimensions": [], "y_range": ()}

    for key in default_hyperparameter_choices:
        if key not in hyperparameters.keys():
            hyperparameters[key] = default_hyperparameter_choices[key]

    return NN(input_dim=input_dim, layers_info=hyperparameters["linear_hidden_units"] + [output_dim],
              output_activation=hyperparameters["final_layer_activation"],
              batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
              hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
              columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
              embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"])


class My_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hyperparameters=None):
        super(My_MLP, self).__init__()
        self.my_nn = create_NN(input_dim, output_dim, hyperparameters)

    def parameters(self):
        return self.my_nn.parameters()

    def forward(self, x, a = None):
        x = x.view(1, -1)
        a = a.view(1, -1)
        x = torch.cat([x, a], dim=1)
        return self.my_nn(x)



class GATNNQ(nn.Module):
    def __init__(self, input_state_dim, input_action_dim, output_dim, mask_dim, emb_dim, window_size, nheads, adj_base):
        super(GATNNQ, self).__init__()
        self.upp = nn.Linear(window_size, emb_dim)
        self.gat1 = GATLayer(emb_dim, nheads, 0)
        self.down = nn.Linear(emb_dim, 1)
        self.out1 = nn.Linear(input_state_dim + input_action_dim, emb_dim)
        self.out2 = nn.Linear(emb_dim, output_dim)
        self.adj = adj_base

    def forward(self, x_state, x_action=None, adj=None):
        if adj is None:
            adj = torch.cat([self.adj for _ in range(x_state.shape[0])], dim=0).to(torch.float32)
        x = self.upp(x_state)
        x = self.gat1(x, adj)
        x = self.down(x)

        x = torch.cat([x, x_action], dim=1)
        x = x.squeeze(-1)
        x = self.out2(F.leaky_relu(self.out1(x)))
        return x


class GATLayer(nn.Module):
    def __init__(self, output_dim, nheads, dropout=0):
        super(GATLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(nheads, output_dim, output_dim // 4, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(output_dim, output_dim, dropout=dropout)

    def forward(self, x, adj):
        x = self.slf_attn(x, adj)
        x = self.pos_ffn(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, dropout=0):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_k)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_k, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        self.gate = GRUGate(d_model)

    def forward(self, x, adj):
        residual = x
        x = self.ln(x)
        q = x
        k = x
        v = x

        d_k, n_head = self.d_k, self.n_head

        # sz_bq, len_qq, qq = q.size()
        sz_b, len_q, _ = q.size()  # 1，21，128
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_k)

        # 矩阵形式转换
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_k)  # (n*b) x lv x dv

        # 对adj按照注意力头进行复制扩充
        adj = adj.unsqueeze(1).repeat(1, n_head, 1, 1).reshape(-1, len_q, len_q)
        output = self.attention(q, k, v, adj)
        output = output.view(n_head, sz_b, len_q, d_k)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = F.relu(self.dropout(self.fc(output)))
        output = self.gate(residual, output)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, dhid, dropout=0):
        super().__init__()
        self.w_1 = nn.Linear(d_in, dhid)
        self.w_2 = nn.Linear(dhid, d_in)
        self.ln = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.gate = GRUGate(d_in)

    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = F.relu(self.w_2(F.relu((self.w_1(x)))))
        x = self.gate(residual, x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, adj):
        attn = torch.bmm(q, k.transpose(1, 2))  # 矩阵相乘
        attn = attn / self.temperature
        adj_mask = (adj == 0)
        attn = attn.masked_fill(adj_mask, -np.inf)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output


class GRUGate(nn.Module):
    def __init__(self, d_model):
        super(GRUGate, self).__init__()

        self.linear_w_r = nn.Linear(d_model, d_model, bias=False)
        self.linear_u_r = nn.Linear(d_model, d_model, bias=False)
        self.linear_w_z = nn.Linear(d_model, d_model)
        self.linear_u_z = nn.Linear(d_model, d_model, bias=False)
        self.linear_w_g = nn.Linear(d_model, d_model, bias=False)
        self.linear_u_g = nn.Linear(d_model, d_model, bias=False)

        self.init_bias()

    def init_bias(self):
        with torch.no_grad():
            self.linear_w_z.bias.fill_(-2)

    def forward(self, x, y):  # x->h y->x
        z = torch.sigmoid(self.linear_w_z(y) + self.linear_u_z(x))
        r = torch.sigmoid(self.linear_w_r(y) + self.linear_u_r(x))
        h_hat = torch.tanh(self.linear_w_g(y) + self.linear_u_g(r * x))
        return (1. - z) * x + z * h_hat
