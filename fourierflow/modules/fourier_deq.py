import math
from functools import partial

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm

from fourierflow.modules.deq.jacobian import jac_loss_estimate
from fourierflow.modules.deq.solvers import anderson, broyden
from fourierflow.registries import Module

from .linear import WNLinear


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
                WNLinear(in_dim, out_dim, wnorm=ff_weight_norm),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers -
                1 else nn.Identity(),
            ))

    def forward(self, x, res):
        for layer in self.layers:
            x = layer(x)
        x = res + x
        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes, factor, norm_locs,
                 ff_weight_norm, n_ff_layers, layer_norm, use_fork, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.norm_locs = norm_locs
        self.gnorm_1 = nn.GroupNorm(out_dim // 16, out_dim)
        self.gnorm_2 = nn.GroupNorm(out_dim // 16, out_dim)

        self.fourier_weight = nn.ParameterList([])
        for _ in range(2):
            weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
            param = nn.Parameter(weight)
            nn.init.xavier_normal_(param, gain=0.1)
            self.fourier_weight.append(param)

        if use_fork:
            self.forecast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
        self.backcast_ff = FeedForward(
            out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, z, x):
        # z.shape == [n_batches, 2 * flat_size, 1]
        # x.shape == [n_batches, flat_size, 1]

        # x.shape == [batch_size, in_dim, grid_size * grid_size]
        M = N = int(math.sqrt(x.shape[1] // self.in_dim))
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        # res.shape == [batch_size, grid_size, grid_size, out_dim]

        x = rearrange(x, 'b (m n i) 1 -> b m n i', m=M, n=N, i=self.in_dim)
        B, M, N, I = x.shape
        # x.shape == [batch_size, grid_size, grid_size, in_dim]

        z = rearrange(z, 'b (m n i) 1 -> b m n i', m=M, n=N, i=I)
        # z.shape == [batch_size, grid_size, grid_size, in_dim]

        backcast = z

        # Subtract away things that we've already used for previous predictions
        x = x + backcast
        # x.shape == [batch_size, grid_size, grid_size, in_dim]

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
            torch.view_as_complex(self.fourier_weight[0]))

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
            torch.view_as_complex(self.fourier_weight[1]))

        xx = torch.fft.irfft(out_ft, dim=-2, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy

        x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        backcast = self.backcast_ff(x, backcast)
        # forecast = self.forecast_ff(x, forecast)

        # out = torch.stack([backcast, forecast], dim=1)
        # out.shape == [batch_size, 2, grid_size, grid_size, out_dim]

        out = rearrange(backcast, 'b m n i -> b (m n i) 1')

        return out


class DEQBlock(nn.Module):
    def __init__(self, modes, width, n_layers, pretraining_steps, norm_locs, factor,
                 ff_weight_norm, n_ff_layers, layer_norm, use_fork, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.pretraining_steps = pretraining_steps
        self.width = width
        self.f = SpectralConv2d(in_dim=width,
                                out_dim=width,
                                n_modes=modes,
                                factor=factor,
                                norm_locs=norm_locs,
                                ff_weight_norm=ff_weight_norm,
                                n_ff_layers=n_ff_layers,
                                layer_norm=layer_norm,
                                use_fork=use_fork,
                                dropout=dropout)
        self.solver = anderson

    def forward(self, z0, x, global_step):
        # z0.shape == [n_batches, width, flat_size]
        if global_step is not None and global_step < self.pretraining_steps:
            z = z0
            for _ in range(self.n_layers):
                z = self.f(z, x)
            return z

        f_thres = 40
        b_thres = 40

        # Forward pass
        with torch.no_grad():
            z_star = self.solver(lambda z: self.f(z, x), z0, threshold=f_thres)[
                'result']   # See step 2 above
            new_z_star = z_star

        # (Prepare for) Backward pass, see step 3 above
        if self.training:
            new_z_star = self.f(z_star.requires_grad_(), x)

            # Jacobian-related computations, see additional step above. For instance:
            # jac_loss = jac_loss_estimate(new_z_star, z_star, vecs=1)

            ggg = autograd.grad(new_z_star, z_star, z_star.new_ones(
                *z_star.shape), retain_graph=True)[0]

            def backward_hook(grad):
                # if self.hook is not None:
                #     self.hook.remove()
                #     torch.cuda.synchronize()   # To avoid infinite recursion
                # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star

                def f(y):
                    return ggg * y + grad

                x0b = torch.zeros_like(grad)
                new_grad = self.solver(f, x0b, b_thres)['result']
                return new_grad

            self.hook = new_z_star.register_hook(backward_hook)

        return new_z_star


@Module.register('fourier_net_2d_deq')
class SimpleBlock2dDEQ(nn.Module):
    def __init__(self, modes, width, input_dim, n_layers, pretraining_steps,
                 norm_locs, n_ff_layers, layer_norm, use_fork, dropout,
                 ff_weight_norm=True, factor=2):
        super().__init__()
        self.width = width
        self.input_dim = input_dim
        self.in_proj = wnorm(nn.Linear(input_dim, self.width), ff_weight_norm)
        self.deq_block = DEQBlock(
            modes, width, n_layers, pretraining_steps, norm_locs, factor,
            ff_weight_norm, n_ff_layers, layer_norm, use_fork, dropout)
        self.out = nn.Sequential(wnorm(nn.Linear(self.width, 128), ff_weight_norm),
                                 wnorm(nn.Linear(128, 1), ff_weight_norm))
        self.solver = broyden
        self.gnorm = nn.GroupNorm(width // 16, width)
        self.norm_locs = norm_locs

    def forward(self, x, global_step=None):
        _, N, M, _ = x.shape
        # x.shape == [n_batches, *dim_sizes, input_size]

        x = self.in_proj(x)
        x = rearrange(x, 'b m n w -> b w m n')
        if 'in' in self.norm_locs:
            x = self.gnorm(x)
        # x.shape == [n_batches, *dim_sizes, width]

        x = rearrange(x, 'b w m n -> b (m n w) 1')
        B, T, _ = x.shape
        # x.shape == [n_batches, flat_size, 1]

        z0 = x.new_zeros([B, T, 1])
        # z0.shape == [n_batches, flat_size, 1]

        new_z_star = self.deq_block(z0, x, global_step)

        new_z_star = rearrange(new_z_star, 'b (m n w) 1 -> b m n w',
                               n=N, m=M, w=self.width)

        forecast = self.out(new_z_star)

        return {
            'forecast': forecast,
        }
