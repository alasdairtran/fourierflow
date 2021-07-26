"""
@author: Zongyi Li This file is the Fourier Neural Operator for 2D problem such
as the Navier-Stokes equation discussed in Section 5.3 in the
[paper](https://arxiv.org/pdf/2010.08895.pdf), which uses a recurrent structure
to propagates in time.
"""


from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fourierflow.common import Module


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes, residual=True, conv_norm=False, nonlinear=False, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.residual = residual
        self.act = nn.ReLU()
        self.act2 = nn.ReLU()
        self.nonlinear = nonlinear
        if residual:
            self.linear = nn.Linear(in_dim, out_dim)
            self.norm = nn.LayerNorm(in_dim) if conv_norm else nn.Identity()
        n = 4 if nonlinear else 2

        fourier_weight = [nn.Parameter(torch.FloatTensor(
            2, in_dim, out_dim, n_modes, 2)) for _ in range(n)]

        self.fourier_weight = nn.ParameterList(fourier_weight)
        for param in self.fourier_weight:
            nn.init.xavier_normal_(param, gain=1/(in_dim*out_dim))

        self.forecast_ff = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim))

        self.backcast_ff = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim))

    @staticmethod
    def complex_matmul_2d(a, b):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)

        b0 = b[0]
        b1 = b[1]

        # (in_channel, out_channel, x), (batch, in_channel, x, y) -> (batch, out_channel, x, y)
        op = partial(torch.einsum, "iox,bixy->boxy")
        c = torch.stack([
            op(b0[..., 0], a[..., 0]) - op(b0[..., 1], a[..., 1]),
            op(b0[..., 1], a[..., 0]) + op(b0[..., 0], a[..., 1])
        ], dim=-1)

        op = partial(torch.einsum, "bixy,ioy->boxy")
        out = torch.stack([
            op(c[..., 0], b1[..., 0]) - op(c[..., 1], b1[..., 1]),
            op(c[..., 1], b1[..., 0]) + op(c[..., 0], b1[..., 1])
        ], dim=-1)

        return out

    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        B, M, N, I = x.shape
        if self.residual:
            res = self.linear(self.norm(x))
        # res.shape == [batch_size, grid_size, grid_size, out_dim]

        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        x_ft = torch.fft.rfft2(x, s=(M, N), norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=4)
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft = torch.zeros(B, I, N, M // 2 + 1, 2, device=x.device)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft[:, :, :self.n_modes, :self.n_modes] = self.complex_matmul_2d(
            x_ft[:, :, :self.n_modes, :self.n_modes], self.fourier_weight[0])

        out_ft[:, :, -self.n_modes:, :self.n_modes] = self.complex_matmul_2d(
            x_ft[:, :, -self.n_modes:, :self.n_modes], self.fourier_weight[1])

        if self.nonlinear:
            out_ft = self.act2(out_ft)
            out_ft[:, :, :self.n_modes, :self.n_modes] = self.complex_matmul_2d(
                x_ft[:, :, :self.n_modes, :self.n_modes], self.fourier_weight[2])

            out_ft[:, :, -self.n_modes:, :self.n_modes] = self.complex_matmul_2d(
                x_ft[:, :, -self.n_modes:, :self.n_modes], self.fourier_weight[3])

        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = torch.fft.irfft2(out_ft, s=(N, M), norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        if self.residual:
            x = self.act(x + res)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x)
        return b, f, [x_ft, out_ft, self.fourier_weight]


@Module.register('fourier_net_2d_split')
class SimpleBlock2dSplit(nn.Module):
    def __init__(self, modes1, modes2, width, input_dim=12, dropout=0.1,
                 n_layers=4, residual=False, add_input=False,
                 nonlinear=False, linear_out: bool = False, weight_sharing: bool = False,
                 norm=False, conv_norm=False, avg_outs=False, next_input='subtract'):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        if isinstance(modes1, list):
            self.modes1 = modes1
        else:
            self.modes1 = [modes1] * n_layers
        self.modes2 = modes2
        self.width = width
        self.in_proj = nn.Linear(input_dim, self.width)
        self.residual = residual
        self.next_input = next_input
        self.avg_outs = avg_outs
        self.weight_sharing = weight_sharing
        self.n_layers = n_layers
        self.add_input = add_input
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        if not weight_sharing:
            self.spectral_layers = nn.ModuleList([])
            for m in self.modes1:
                self.spectral_layers.append(SpectralConv2d(in_dim=width,
                                                           out_dim=width,
                                                           n_modes=m,
                                                           residual=residual,
                                                           nonlinear=nonlinear,
                                                           conv_norm=conv_norm,
                                                           dropout=dropout))
        else:
            self.layer = SpectralConv2d(in_dim=width,
                                        out_dim=width,
                                        n_modes=modes1,
                                        residual=residual,
                                        nonlinear=nonlinear,
                                        conv_norm=conv_norm,
                                        dropout=dropout)

        self.out = nn.Sequential(
            nn.LayerNorm(self.width) if norm else nn.Identity(),
            nn.Linear(self.width, 128),
            nn.Identity() if linear_out else nn.ReLU(),
            nn.Linear(128, 1))

    def forward(self, x):
        # x.shape == [n_batches, *dim_sizes, input_size]
        forecast = x[..., 0:1] if self.add_input else 0
        x = self.in_proj(x)
        forecast_list = []
        out_fts = []
        for i in range(self.n_layers):
            layer = self.layer if self.weight_sharing else self.spectral_layers[i]
            b, f, out_ft = layer(x)
            out_fts.append(out_ft)
            f_out = self.out(f)
            forecast = forecast + f_out
            forecast_list.append(f_out)
            if self.next_input == 'subtract':
                x = x - b
            elif self.next_input == 'b':
                x = b
            elif self.next_input == 'f':
                x = f
            elif self.next_input == 'add':
                x = x + b
            else:
                raise ValueError

        if self.avg_outs:
            forecast = forecast / len(self.spectral_layers)

        return forecast, forecast_list, out_fts
