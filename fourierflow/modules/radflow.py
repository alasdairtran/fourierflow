import torch.nn as nn
import torch.nn.functional as F

from fourierflow.common import Module

from .linear import GehringLinear


@Module.register('radflow_lstm')
class LSTMDecoder(Module):
    def __init__(self, hidden_size, n_layers, dropout, input_size):
        super().__init__()
        self.input_size = input_size
        self.in_proj = GehringLinear(input_size, hidden_size)
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            self.layers.append(LSTMLayer(hidden_size, dropout))

        self.out_f = GehringLinear(hidden_size, input_size)

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
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.layer = nn.LSTM(hidden_size, hidden_size, 1,
                             batch_first=True)
        self.drop = nn.Dropout(dropout)

        self.proj_f = GehringLinear(hidden_size, hidden_size)
        self.proj_b = GehringLinear(hidden_size, hidden_size)
        self.out_f = GehringLinear(hidden_size, hidden_size)
        self.out_b = GehringLinear(hidden_size, hidden_size)

    def forward(self, X):
        # X.shape == [batch_size, seq_len]
        # yn.shape == [batch_size, n_neighs, seq_len]

        # It's recommended to apply dropout on the input to the LSTM cell
        # See https://ieeexplore.ieee.org/document/7333848
        X = self.drop(X)

        X, _ = self.layer(X)
        # X.shape == [batch_size, seq_len, hidden_size]

        b = self.out_b(F.gelu(self.proj_b(X)))
        f = self.out_f(F.gelu(self.proj_f(X)))
        # b.shape == f.shape == [batch_size, seq_len, hidden_size]

        return b, f
