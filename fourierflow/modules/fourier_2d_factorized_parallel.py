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
from torch.nn.utils import weight_norm

from fourierflow.common import Module


def wnorm(module, active):
    return weight_norm(module) if active else module


class FeedForward(nn.Module):
    def __init__(self, dim, factor, ff_weight_norm):
        super().__init__()
        self.linear_1 = wnorm(nn.Linear(dim, dim * factor), ff_weight_norm)
        self.act = nn.ReLU(inplace=True)
        self.linear_2 = wnorm(nn.Linear(dim * factor, dim), ff_weight_norm)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes, forecast_ff, backcast_ff,
                 fourier_weight, factor, norm_locs, group_width, ff_weight_norm):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.norm_locs = norm_locs
        self.group_width = group_width

        self.fourier_weight = fourier_weight
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(2):
                weight = torch.complex(
                    torch.FloatTensor(in_dim, out_dim, n_modes),
                    torch.FloatTensor(in_dim, out_dim, n_modes))
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        self.forecast_ff = forecast_ff
        if not self.forecast_ff:
            self.forecast_ff = FeedForward(out_dim, factor, ff_weight_norm)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(out_dim, factor, ff_weight_norm)

    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        B, M, N, I = x.shape

        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_fty.new_zeros(B, I, N, M // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft[:, :, :, :self.n_modes] = torch.einsum(
            "bixy,ioy->boxy",
            x_fty[:, :, :, :self.n_modes],
            self.fourier_weight[0])

        xy = torch.fft.irfft(out_ft, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, N // 2 + 1, M)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :self.n_modes, :] = torch.einsum(
            "bixy,iox->boxy",
            x_ftx[:, :, :self.n_modes, :],
            self.fourier_weight[1])

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
                 n_layers=4, linear_out: bool = False, share_weight: bool = False,
                 avg_outs=False, next_input='subtract', share_fork=False, factor=2,
                 norm_locs=[], group_width=16, ff_weight_norm=False):
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
        self.in_proj = wnorm(nn.Linear(input_dim, self.width), ff_weight_norm)
        self.next_input = next_input
        self.avg_outs = avg_outs
        self.n_layers = n_layers
        self.norm_locs = norm_locs
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.forecast_ff = self.backcast_ff = None
        if share_fork:
            self.forecast_ff = FeedForward(width, factor, ff_weight_norm)
            self.backcast_ff = FeedForward(width, factor, ff_weight_norm)

        self.fourier_weight = None
        if share_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(2):
                weight = torch.complex(torch.FloatTensor(width, width, modes),
                                       torch.FloatTensor(width, width, modes))
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv2d(in_dim=width,
                                                       out_dim=width,
                                                       n_modes=modes,
                                                       forecast_ff=self.forecast_ff,
                                                       backcast_ff=self.backcast_ff,
                                                       fourier_weight=self.fourier_weight,
                                                       factor=factor,
                                                       norm_locs=norm_locs,
                                                       group_width=group_width,
                                                       ff_weight_norm=ff_weight_norm))

        self.out = nn.Sequential(
            wnorm(nn.Linear(self.width, 128), ff_weight_norm),
            wnorm(nn.Linear(128, 1), ff_weight_norm))

    def forward(self, x):
        # x.shape == [n_batches, *dim_sizes, input_size]
        forecast = 0
        x = self.in_proj(x)
        forecast_list = []
        out_fts = []
        for i in range(self.n_layers):
            layer = self.spectral_layers[i]
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
