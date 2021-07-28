"""
@author: Zongyi Li This file is the Fourier Neural Operator for 2D problem such
as the Navier-Stokes equation discussed in Section 5.3 in the
[paper](https://arxiv.org/pdf/2010.08895.pdf), which uses a recurrent structure
to propagates in time.
"""


import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fourierflow.common import Module


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.act = nn.ReLU()
        self.act2 = nn.ReLU()

        fourier_weight = [nn.Parameter(torch.FloatTensor(
            in_dim, out_dim, n_modes, 2)) for _ in range(2)]

        self.fourier_weight = nn.ParameterList(fourier_weight)
        for param in self.fourier_weight:
            nn.init.xavier_normal_(param)

        self.forecast_ff = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim))

        self.backcast_ff = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim))

    @staticmethod
    def complex_matmul_y_2d(a, b):
        op = partial(torch.einsum, "bixy,ioy->boxy")
        out = torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)
        return out

    @staticmethod
    def complex_matmul_x_2d(a, b):
        op = partial(torch.einsum, "bixy,iox->boxy")
        out = torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)
        return out

    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        B, M, N, I = x.shape

        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        x_fty = torch.stack([x_fty.real, x_fty.imag], dim=4)
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft = torch.zeros(B, I, N, M // 2 + 1, 2, device=x.device)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft[:, :, :, :self.n_modes] = self.complex_matmul_y_2d(
            x_fty[:, :, :, :self.n_modes], self.fourier_weight[0])

        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        xy = torch.fft.irfft(out_ft, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        x_ftx = torch.stack([x_ftx.real, x_ftx.imag], dim=4)
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft = torch.zeros(B, I, N // 2 + 1, M, 2, device=x.device)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :self.n_modes, :] = self.complex_matmul_x_2d(
            x_ftx[:, :, :self.n_modes, :], self.fourier_weight[1])

        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        xx = torch.fft.irfft(out_ft, dim=-2, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy

        x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        b = self.backcast_ff(x)
        f = self.forecast_ff(x)
        return b, f, []


@Module.register('fourier_2d_factorized_parallel')
class SimpleBlock2dFactorizedParallel(nn.Module):
    def __init__(self, modes, width, input_dim=12, dropout=0.1,
                 n_layers=4, linear_out: bool = False, weight_sharing: bool = False,
                 avg_outs=False, next_input='subtract'):
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

        self.modes = modes
        self.width = width
        self.in_proj = nn.Linear(input_dim, self.width)
        self.next_input = next_input
        self.avg_outs = avg_outs
        self.weight_sharing = weight_sharing
        self.n_layers = n_layers
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        if not weight_sharing:
            self.spectral_layers = nn.ModuleList([])
            for _ in range(n_layers):
                self.spectral_layers.append(SpectralConv2d(in_dim=width,
                                                           out_dim=width,
                                                           n_modes=modes))
        else:
            self.layer = SpectralConv2d(in_dim=width,
                                        out_dim=width,
                                        n_modes=modes)

        self.out = nn.Sequential(nn.Linear(self.width, 128),
                                 nn.Linear(128, 1))

    def forward(self, x):
        # x.shape == [n_batches, *dim_sizes, input_size]
        forecast = 0
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
            elif self.next_input == 'add':
                x = x + b

        if self.avg_outs:
            forecast = forecast / len(self.spectral_layers)

        return forecast, forecast_list, out_fts
