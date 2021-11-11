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

from fourierflow.registries import Module


def wnorm(module, active):
    return weight_norm(module) if active else module


class FeedForward(nn.Module):
    def __init__(self, dim, factor, ff_weight_norm, n_layers, layer_norm, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(nn.Sequential(
                wnorm(nn.Linear(in_dim, out_dim), ff_weight_norm),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers -
                1 else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SpectralConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes,
                 fourier_weight, factor, norm_locs, group_width, ff_weight_norm,
                 n_ff_layers, layer_norm, dropout, mode):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.norm_locs = norm_locs
        self.group_width = group_width
        self.mode = mode
        self.fourier_weight = fourier_weight
        self.backcast_ff = FeedForward(
            out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, xs, orders):
        # x.shape == [batch_size, n_cells, in_dim]
        B, C, I = xs.shape

        outs = []
        for o, shape in enumerate('DUNE'):
            order = orders[shape]
            x = xs[:, order['p']]
            x = rearrange(x, 'b m i -> b i m')
            # x.shape == [batch_size, in_dim, grid_size, grid_size]

            # # # Dimesion Y # # #
            x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
            # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

            out_ft = x_fty.new_zeros(B, I, C // 2 + 1)
            # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

            if self.mode == 'full':
                out_ft[:, :, :self.n_modes] = torch.einsum(
                    "bix,iox->box",
                    x_fty[:, :, :self.n_modes],
                    torch.view_as_complex(self.fourier_weight[o]))
            elif self.mode == 'low-pass':
                out_ft[:, :, :self.n_modes] = x_fty[:, :, :self.n_modes]

            x = torch.fft.irfft(out_ft, n=C, dim=-1, norm='ortho')
            # x.shape == [batch_size, in_dim, grid_size]

            outs.append(x[:, :, order['r']])

        x = outs[0] + outs[1] + outs[2] + outs[3]
        x = rearrange(x, 'b i m -> b m i')
        # x.shape == [batch_size, grid_size, out_dim]

        b = self.backcast_ff(x)
        # f = self.forecast_ff(x)
        return b, None, []


@Module.register('fourier_1d_factorized')
class SimpleBlock1dFactorized(nn.Module):
    def __init__(self, modes, width, input_dim=12, dropout=0.0, in_dropout=0.0,
                 n_layers=4, linear_out: bool = False, share_weight: bool = False,
                 avg_outs=False, next_input='subtract', share_fork=False, factor=2,
                 norm_locs=[], group_width=16, ff_weight_norm=False, n_ff_layers=2,
                 gain=1, layer_norm=False, mode='full', output_size=2):
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
        self.input_dim = input_dim
        self.in_proj = wnorm(nn.Linear(input_dim, self.width), ff_weight_norm)
        self.drop = nn.Dropout(in_dropout)
        self.next_input = next_input
        self.avg_outs = avg_outs
        self.n_layers = n_layers
        self.norm_locs = norm_locs
        self.orders = None
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.fourier_weight = nn.ParameterList([])
        for _ in range(4):
            weight = torch.FloatTensor(width, width, modes, 2)
            param = nn.Parameter(weight)
            nn.init.xavier_normal_(param, gain=gain)
            self.fourier_weight.append(param)

        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv1d(in_dim=width,
                                                       out_dim=width,
                                                       n_modes=modes,
                                                       fourier_weight=self.fourier_weight,
                                                       factor=factor,
                                                       norm_locs=norm_locs,
                                                       group_width=group_width,
                                                       ff_weight_norm=ff_weight_norm,
                                                       n_ff_layers=n_ff_layers,
                                                       layer_norm=layer_norm,
                                                       dropout=dropout,
                                                       mode=mode))

        self.out = nn.Sequential(
            wnorm(nn.Linear(self.width, 128), ff_weight_norm),
            wnorm(nn.Linear(128, output_size), ff_weight_norm))

    def register_orders(self, orders):
        self.orders = orders

    def forward(self, x, **kwargs):
        B, M, N, I = x.shape
        # x.shape == [n_batches, *dim_sizes, input_size]

        x = rearrange(x, 'b m n i -> b (m n) i')
        # x.shape == [batch_size, n_cells, in_dim]

        forecast = 0
        x = self.in_proj(x)
        # temp = x
        x = self.drop(x)
        forecast_list = []
        out_fts = []
        for i in range(self.n_layers):
            layer = self.spectral_layers[i]
            b, _, out_ft = layer(x, self.orders)
            # b, _, out_ft = layer(temp)
            out_fts.append(out_ft)
            # f_out = self.out(f)
            # forecast = forecast + f_out
            # forecast_list.append(f_out)
            if self.next_input == 'subtract':
                x = x - b
            elif self.next_input == 'add':
                x = x + b
                # temp = x + b

        forecast = self.out(b)
        if self.avg_outs:
            forecast = forecast / len(self.spectral_layers)

        forecast = rearrange(forecast, 'b (m n) i -> b m n i', m=M, n=N)

        return {
            'forecast': forecast,
            'forecast_list': forecast_list,
        }
