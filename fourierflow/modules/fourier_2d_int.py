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

from fourierflow.registries import Module

# from torchdiffeq import odeint


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes, resdiual=True, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.linear = nn.Linear(in_dim, out_dim)
        self.residual = resdiual
        self.act = nn.ReLU(inplace=True)
        n = 2

        fourier_weight = [nn.Parameter(torch.FloatTensor(
            in_dim, out_dim, n_modes, n_modes, 2)) for _ in range(n)]
        self.fourier_weight = nn.ParameterList(fourier_weight)
        for param in self.fourier_weight:
            nn.init.xavier_normal_(param, gain=1/(in_dim*out_dim))

        self.forecast_ff = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim))

        self.backcast_ff = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim))

    @staticmethod
    def complex_matmul_2d(a, b):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        op = partial(torch.einsum, "bixy,ioxy->boxy")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        B, M, N, I = x.shape
        res = self.linear(x)
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

        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = torch.fft.irfft2(out_ft, s=(N, M), norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        if self.residual:
            x = self.act(x + res)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x)
        return b, f


@Module.register('fourier_net_2d_integrate')
class SimpleBlock2dIntegrate(nn.Module):
    def __init__(self, modes1, modes2, width, input_dim=12, dropout=0.1,
                 n_layers=4, residual=False, conv_residual=True,
                 linear_out: bool = False,
                 norm=False):
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

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_proj = nn.Linear(input_dim, self.width)
        self.residual = residual
        self.n_layers = n_layers
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.f = SpectralConv2d(in_dim=width,
                                out_dim=width,
                                n_modes=modes1,
                                resdiual=conv_residual,
                                dropout=dropout)

        self.out = nn.Sequential(
            nn.LayerNorm(self.width) if norm else nn.Identity(),
            nn.Linear(self.width, 128),
            nn.Identity() if linear_out else nn.ReLU(inplace=True),
            nn.Linear(128, 1))

    def forward(self, x):
        # x.shape == [n_batches, *dim_sizes, input_size]
        x = self.in_proj(x)

        # ts = x.new_tensor([0, 1])
        # x = odeint(self.f, x, ts, method='dopri5')[-1]
        forecast = 0
        for i in range(self.n_layers):
            b, f = self.f(x)
            x = x - b
            forecast = forecast + f

        forecast = self.out(forecast)

        return forecast, None
