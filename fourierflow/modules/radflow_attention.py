import torch
import torch.nn as nn
import torch.nn.functional as F

from fourierflow.registry import Module

from .attention import Attention
from .rotary import SinusoidalEmbeddings


@Module.register('radflow_attention')
class AttentionDecoder(Module):
    def __init__(self, hidden_size, n_layers, dropout, input_size):
        super().__init__()
        self.input_size = input_size
        self.in_proj = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            self.layers.append(LSTMLayer(hidden_size, dropout))

        self.out_f = nn.Linear(hidden_size, input_size)

    def forward(self, X):
        # X.shape == [batch_size, seq_len]
        # yn.shape == [batch_size, n_neighs, seq_len]

        if len(X.shape) == 2:
            X = X.unsqueeze(-1)
        # X.shape == [batch_size, seq_len, 1]

        X = self.in_proj(X)
        # X.shape == [batch_size, seq_len, hidden_size]

        forecast = X.new_zeros(*X.shape)
        for layer in self.layers:
            b, f = layer(X)
            X = X - b
            forecast = forecast + f

        f = self.out_f(forecast)

        if self.input_size == 1:
            f = f.squeeze(-1)

        return f


class LSTMLayer(nn.Module):
    def __init__(self, hidden_size, dropout, dim_head=64, heads=4):
        super().__init__()
        self.attn = Attention(hidden_size, heads=heads,
                              dim_head=dim_head, dropout=dropout)
        self.drop = nn.Dropout(dropout)

        self.proj_f = nn.Linear(hidden_size, hidden_size)
        self.proj_b = nn.Linear(hidden_size, hidden_size)
        self.out_f = nn.Linear(hidden_size, hidden_size)
        self.out_b = nn.Linear(hidden_size, hidden_size)

        self.sinu_emb = SinusoidalEmbeddings(dim_head, max_ts=100)

    def forward(self, X):
        b, n, _ = X.shape
        # X.shape == [batch_size, seq_len]
        # yn.shape == [batch_size, n_neighs, seq_len]

        # It's recommended to apply dropout on the input to the LSTM cell
        # See https://ieeexplore.ieee.org/document/7333848
        X = self.drop(X)

        # bottom right triangle is filled with True
        pos_emb = self.sinu_emb(X)
        mask = torch.tril(X.new_ones(n, n)) == 1
        X = self.attn(X, mask=mask, pos_emb=pos_emb)
        # X.shape == [batch_size, seq_len, hidden_size]

        b = self.out_b(F.gelu(self.proj_b(X)))
        f = self.out_f(F.gelu(self.proj_f(X)))
        # b.shape == f.shape == [batch_size, seq_len, hidden_size]

        return b, f
