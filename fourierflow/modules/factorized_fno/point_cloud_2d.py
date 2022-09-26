"""
@author: Zongyi Li and Daniel Zhengyu Huang
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .grid_2d import SpectralConv2d as FactorizedSpectralConv2d


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, s1=32, s2=32, transform=True):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.s1 = s1
        self.s2 = s2

        if transform:
            self.scale = (1 / (in_channels * out_channels))
            self.weights1 = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None, transform=True):
        batchsize = u.shape[0]

        # Compute Fourier coefficients up to factor of e^(- something constant)
        if x_in is None:
            u_ft = torch.fft.rfft2(u)
            s1 = u.size(-2)
            s2 = u.size(-1)
        else:
            u_ft = self.fft2d(u, x_in, iphi, code)
            s1 = self.s1
            s2 = self.s2

        # Multiply relevant Fourier modes
        # print(u.shape, u_ft.shape)
        if transform:
            factor1 = self.compl_mul2d(
                u_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            factor2 = self.compl_mul2d(
                u_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        else:
            factor1 = u_ft[:, :, :self.modes1, :self.modes2]
            factor2 = u_ft[:, :, -self.modes1:, :self.modes2]

        # Return to physical space
        if x_out == None:
            out_ft = torch.zeros(batchsize, self.out_channels, s1,
                                 s2 // 2 + 1, dtype=torch.cfloat, device=u.device)
            out_ft[:, :, :self.modes1, :self.modes2] = factor1
            out_ft[:, :, -self.modes1:, :self.modes2] = factor2
            u = torch.fft.irfft2(out_ft, s=(s1, s2))
        else:
            out_ft = torch.cat([factor1, factor2], dim=-2)
            u = self.ifft2d(out_ft, x_out, iphi, code)

        return u

    def fft2d(self, u, x_in, iphi=None, code=None):
        # u (batch, channels, n)
        # x_in (batch, n_points, 2) locations in [0,1]*[0,1]
        # iphi: function: x_in -> x_c

        B = x_in.shape[0]
        N = x_in.shape[1]
        device = x_in.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 = torch.cat((torch.arange(start=0, end=self.modes1, step=1),
                          torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1, 1).repeat(1, m2).to(device)
        k_x2 = torch.cat((torch.arange(start=0, end=self.modes2, step=1),
                          torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1, m2).repeat(m1, 1).to(device)

        # Shift the mesh coords into the right location on the unit square.
        if iphi is None:
            x = x_in
        else:
            x = iphi(x_in, code)

        # x.shape == [B, N, 2]
        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[..., 0].view(-1), k_x1.view(-1)
                         ).reshape(B, N, m1, m2)
        K2 = torch.outer(x[..., 1].view(-1), k_x2.view(-1)
                         ).reshape(B, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(-1j * 2 * np.pi * K).to(device)

        # Y (batch, channels, N)
        u = u + 0j
        Y = torch.einsum("bcn,bnxy->bcxy", u, basis)
        return Y

    def ifft2d(self, u_ft, x_out, iphi=None, code=None):
        # u_ft (batch, channels, kmax, kmax)
        # x_out (batch, N, 2) locations in [0,1]*[0,1]
        # iphi: function: x_out -> x_c

        batchsize = x_out.shape[0]
        N = x_out.shape[1]
        device = x_out.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 = torch.cat((torch.arange(start=0, end=self.modes1, step=1),
                          torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1, 1).repeat(1, m2).to(device)
        k_x2 = torch.cat((torch.arange(start=0, end=self.modes2, step=1),
                          torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1, m2).repeat(m1, 1).to(device)

        if iphi == None:
            x = x_out
        else:
            x = iphi(x_out, code)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[:, :, 0].view(-1), k_x1.view(-1)
                         ).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[:, :, 1].view(-1), k_x2.view(-1)
                         ).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(1j * 2 * np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
        u_ft = torch.cat([u_ft, u_ft2], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", u_ft, basis)
        Y = Y.real
        return Y


class FNOFactorizedPointCloud2D(nn.Module):
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
            if i == 0:
                conv = SpectralConv2d(
                    self.width, self.width, self.modes1, self.modes2, s1, s2, transform=False)
            elif i == self.n_layers:
                conv = SpectralConv2d(
                    self.width, self.width, self.modes1, self.modes2, s1, s2)
            else:
                conv = FactorizedSpectralConv2d(in_dim=width,
                                                out_dim=width,
                                                n_modes=modes1,
                                                forecast_ff=None,
                                                backcast_ff=None,
                                                fourier_weight=None,
                                                factor=2,
                                                ff_weight_norm=True,
                                                n_ff_layers=2,
                                                layer_norm=False,
                                                use_fork=False,
                                                dropout=0.0,
                                                mode='full')
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
                            code=code, transform=False)  # [20, 32, 40, 40]
        uc3 = self.bs[0](grid)
        uc = uc1 + uc3

        # uc.shape == [20, 32, 40, 40]
        for i in range(1, self.n_layers):
            uc1 = rearrange(uc, 'b c h w -> b h w c')
            uc1 = self.convs[i](uc1)[0]
            uc1 = rearrange(uc1, 'b h w c -> b c h w')
            # uc2 = self.ws[i-1](uc)
            uc3 = self.bs[0](grid)
            uc = uc + uc1 + uc3

        L = self.n_layers
        u = self.convs[L](uc, x_out=x_out, iphi=iphi, code=code)
        u3 = self.bs[-1](x_out.permute(0, 2, 1))
        u = u + u3

        u = u.permute(0, 2, 1)
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
