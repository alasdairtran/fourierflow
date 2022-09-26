"""
@author: Zongyi Li and Daniel Zhengyu Huang
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..feedforward import FeedForward
from ..linear import WNLinear


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, m1, m2, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, dropout, s1, s2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.m1 = m1
        self.m2 = m2
        self.s1 = s1
        self.s2 = s2

        self.fourier_weight = fourier_weight
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [m2, m1]:
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x, x_in=None, x_out=None, iphi=None, code=None, ff=True):
        x = self.forward_fourier(x, x_in, x_out, iphi, code)
        if ff:
            x = rearrange(x, 'b i m n -> b m n i')
            x = self.backcast_ff(x)
            x = rearrange(x, 'b m n i -> b i m n')
        return x

    def forward_fourier(self, x, x_in, x_out, iphi, code):
        # Compute the basis if needed
        if x_in is not None:
            basis_fft_x, basis_fft_y = self.get_fft_bases(x_in, iphi, code)
        if x_out is not None:
            basis_ifft_x, basis_ifft_y = self.get_ifft_bases(x_out, iphi, code)
        # basis_x.shape == [batch_size, n_points, m1, s2]
        # basis_y.shape == [batch_size, n_points, s1, m2]

        # # # Dimesion Y # # #
        if x_in is None:
            # x.shape == [batch_size, hidden_size, s1, s2]
            x_fty = torch.fft.rfft(x, dim=-1)
            # x_fty.shape == [batch_size, hidden_size, s1, s2 // 2 + 1]
        else:
            # x.shape == [batch_size, hidden_size, n_points]
            x_fty = torch.einsum("bcn,bnxy->bcxy", x + 0j, basis_fft_y)
            # x_fty.shape == [batch_size, hidden_size, s1, m2]

        B, H = x_fty.shape[:2]
        out_ft = x_fty.new_zeros(B, H, self.s1, self.s2 // 2 + 1)
        # out_ft.shape == [batch_size, hidden_size, s1, s2 // 2 + 1]

        out_ft[:, :, :, :self.m2] = torch.einsum(
            "bixy,ioy->boxy",
            x_fty[:, :, :, :self.m2],
            torch.view_as_complex(self.fourier_weight[0]))

        if x_out is None:
            xy = torch.fft.irfft(out_ft, n=self.s1, dim=-1)
            # xy.shape == [batch_size, in_dim, grid_size, grid_size]
        else:
            # out_ft.shape == [batch_size, hidden_size, s1, m2]

            xy = torch.einsum("bcxy,bnxy->bcn",
                              out_ft[:, :, :, :self.m2], basis_ifft_y).real
            # xy.shape == [batch_size, in_dim, n_points]

        # # # Dimesion X # # #
        if x_in is None:
            x_ftx = torch.fft.rfft(x, dim=-2)
            # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]
        else:
            # x.shape == [batch_size, hidden_size, n_points]
            x_ftx = torch.einsum("bcn,bnxy->bcxy", x + 0j, basis_fft_x)
            # x_ftx.shape == [batch_size, hidden_size, m1, s2]

        B, H = x_ftx.shape[:2]
        out_ft = x_ftx.new_zeros(B, H, self.s1 // 2 + 1, self.s2)
        # out_ft.shape == [batch_size, hidden_size, s1 // 2 + 1, s2]

        out_ft[:, :, :self.m1, :] = torch.einsum(
            "bixy,iox->boxy",
            x_ftx[:, :, :self.m1, :],
            torch.view_as_complex(self.fourier_weight[1]))

        if x_out is None:
            xx = torch.fft.irfft(out_ft, n=self.s2, dim=-2)
            # xx.shape == [batch_size, in_dim, grid_size, grid_size]
        else:
            xx = torch.einsum("bcxy,bnxy->bcn",
                              out_ft[:, :, :self.m1, :], basis_ifft_x).real
            # xy.shape == [batch_size, in_dim, n_points]

        # # Combining Dimensions # #
        x = xx + xy

        return x

    def get_fft_bases(self, mesh_coords, iphi, code):
        device = self.fourier_weight[0].device

        k_x1 = torch.arange(0, self.m1).to(device)
        k_x2 = torch.arange(0, self.m2).to(device)
        x = iphi(mesh_coords, code)  # [20, 972, 2]

        B, N, _ = x.shape
        K1 = torch.outer(x[..., 1].view(-1), k_x1)
        K1 = K1.reshape(B, N, self.m1)
        # K1.shape == [batch_size, n_points, m1]

        K2 = torch.outer(x[..., 0].view(-1), k_x2)
        K2 = K2.reshape(B, N, self.m2)
        # K2.shape == [batch_size, n_points, m2]

        basis_x = torch.exp(-1j * 2 * np.pi * K1).to(device)
        basis_x = repeat(basis_x, "b n m1 -> b n m1 s2", s2=self.s2)
        # basis_x.shape == [batch_size, n_points, m1, s2]

        basis_y = torch.exp(-1j * 2 * np.pi * K2).to(device)
        basis_y = repeat(basis_y, "b n m2 -> b n s1 m2", s1=self.s1)
        # basis_y.shape == [batch_size, n_points, s1, m2]

        return basis_x, basis_y

    def get_ifft_bases(self, mesh_coords, iphi, code):
        device = self.fourier_weight[0].device

        k_x1 = torch.arange(0, self.m1).to(device)
        k_x2 = torch.arange(0, self.m2).to(device)
        x = iphi(mesh_coords, code)  # [20, 972, 2]

        B, N, _ = x.shape
        K1 = torch.outer(x[..., 1].view(-1), k_x1)
        K1 = K1.reshape(B, N, self.m1)
        # K1.shape == [batch_size, n_points, m1]

        K2 = torch.outer(x[..., 0].view(-1), k_x2)
        K2 = K2.reshape(B, N, self.m2)
        # K2.shape == [batch_size, n_points, m2]

        basis_x = torch.exp(1j * 2 * np.pi * K1).to(device)
        basis_x = repeat(basis_x, "b n m1 -> b n m1 s2", s2=self.s2)
        # basis_x.shape == [batch_size, n_points, m1, s2]

        basis_y = torch.exp(1j * 2 * np.pi * K2).to(device)
        basis_y = repeat(basis_y, "b n m2 -> b n s1 m2", s1=self.s1)
        # basis_y.shape == [batch_size, n_points, s1, m2]

        return basis_x, basis_y


class FNOFullyFactorizedMesh2D(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, out_channels,
                 n_layers=4, is_mesh=True, s1=40, s2=40):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.is_mesh = is_mesh
        self.s1 = s1
        self.s2 = s2
        self.n_layers = n_layers

        # input channel is 3: (a(x, y), x, y)
        self.fc0 = nn.Linear(in_channels, self.width)

        self.convs = nn.ModuleList([])
        self.ws = nn.ModuleList([])
        self.bs = nn.ModuleList([])

        # if factorized:
        #     self.fourier_weight = nn.ParameterList([])
        #     for _ in range(2):
        #         weight = torch.FloatTensor(width, width, modes1, 2)
        #         param = nn.Parameter(weight)
        #         nn.init.xavier_normal_(param, gain=1)
        #         self.fourier_weight.append(param)

        for i in range(self.n_layers + 1):
            conv = SpectralConv2d(in_dim=width,
                                  out_dim=width,
                                  m1=modes1,
                                  m2=modes2,
                                  backcast_ff=None,
                                  fourier_weight=None,
                                  factor=2,
                                  ff_weight_norm=True,
                                  n_ff_layers=2,
                                  layer_norm=False,
                                  dropout=0.0,
                                  s1=s1,
                                  s2=s2)
            self.convs.append(conv)

        self.bs.append(nn.Conv2d(2, self.width, 1))
        self.bs.append(nn.Conv1d(2, self.width, 1))

        for i in range(self.n_layers - 1):
            w = nn.Conv2d(self.width, self.width, 1)
            self.ws.append(w)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, u, code=None, x_in=None, x_out=None, iphi=None):
        # u.shape == [batch_size, n_points, 2] are the coords.
        # code.shape == [batch_size, 42] are the input features
        # x_in (batch, Nx, 2) the input mesh (sampling mesh)
        # xi (batch, xi1, xi2, 2) the computational mesh (uniform)
        # x_in (batch, Nx, 2) the input mesh (query mesh)

        if self.is_mesh and x_in is None:
            x_in = u
        if self.is_mesh and x_out is None:
            x_out = u

        # grid is like the (x, y) coordinates of a unit square [0, 1]^2
        grid = self.get_grid([u.shape[0], self.s1, self.s2],
                             u.device).permute(0, 3, 1, 2)
        # grid.shape == [batch_size, 2, size_x, size_y] == [20, 2, 40, 40]
        # grid[:, 0, :, :] is the row index (y-coordinate)
        # grid[:, 1, :, :] is the column index (x-coordinate)

        # Projection to higher dimension
        u = self.fc0(u)
        u = u.permute(0, 2, 1)
        # u.shape == [batch_size, hidden_size, n_points]

        uc1 = self.convs[0](u, x_in=x_in, iphi=iphi,
                            code=code)  # [B, H, S1, S2]
        uc3 = self.bs[0](grid)  # [B, H, S1, S2]
        uc = uc1 + uc3  # [B, H, S1, S2]

        # [B, H, S1, S2]
        for i in range(1, self.n_layers):
            uc1 = self.convs[i](uc)
            uc3 = self.bs[0](grid)
            uc = uc + uc1 + uc3

        L = self.n_layers
        u = self.convs[L](uc, x_out=x_out, iphi=iphi, code=code, ff=False)
        # u.shape == [B, H, N]
        u3 = self.bs[-1](x_out.permute(0, 2, 1))
        u = u + u3

        u = u.permute(0, 2, 1)
        # u.shape == [B, N, H]
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(
            [batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(
            [batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
