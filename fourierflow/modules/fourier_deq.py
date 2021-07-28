import math
from functools import partial

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fourierflow.common import Module
from fourierflow.modules.deq.jacobian import jac_loss_estimate
from fourierflow.modules.deq.solvers import anderson, broyden


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.0, weight_norm=False):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim * 2)
        self.act = nn.ReLU()
        self.linear_2 = nn.Linear(dim * 2, dim)
        self.weight_norm = weight_norm
        self.dropout = dropout
        # self.reset_parameters()
        self.gnorm = nn.GroupNorm(dim // 16, dim)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.linear_1.weight)
        nn.init.xavier_normal_(self.linear_2.weight)

        # Weight normalization is a reparameterization that decouples the
        # magnitude of a weight tensor from its direction. See Salimans and
        # Kingma (2016): https://arxiv.org/abs/1602.07868.
        if self.weight_norm:
            nn.utils.weight_norm(self.linear_1)
            nn.utils.weight_norm(self.linear_2)

    def forward(self, x, res):
        x = self.linear_2(self.act(self.linear_1(x)))
        x = res + x
        x = rearrange(x, 'b m n i -> b i m n')
        x = self.gnorm(x)
        x = rearrange(x, 'b i m n -> b m n i')
        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes, size):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.act = nn.ReLU()
        self.act2 = nn.ReLU()

        fourier_weight = [nn.Parameter(torch.FloatTensor(
            in_dim, out_dim, n_modes, 2)) for _ in range(4)]

        self.fourier_weight = nn.ParameterList(fourier_weight)
        for param in self.fourier_weight:
            nn.init.xavier_normal_(param)

        self.forecast_ff = FeedForward(out_dim)
        self.backcast_ff = FeedForward(out_dim)

    @staticmethod
    def complex_matmul_2d(a, b0, b1):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)

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

        z = rearrange(z, 'b (k m n i) 1 -> k b m n i', k=2, m=M, n=N, i=I)
        # z.shape == [2, batch_size, grid_size, grid_size, in_dim]

        backcast, forecast = z[0], z[1]

        # Subtract away things that we've already used for previous predictions
        x = x - backcast
        # x.shape == [batch_size, grid_size, grid_size, in_dim]

        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        x_ft = torch.fft.rfft2(x, s=(M, N), norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size // 2 + 1]

        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=4)
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft = torch.zeros(B, I, N, M // 2 + 1, 2, device=x.device)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft[:, :, :self.n_modes, :self.n_modes] = self.complex_matmul_2d(
            x_ft[:, :, :self.n_modes, :self.n_modes], self.fourier_weight[0], self.fourier_weight[1])

        out_ft[:, :, -self.n_modes:, :self.n_modes] = self.complex_matmul_2d(
            x_ft[:, :, -self.n_modes:, :self.n_modes], self.fourier_weight[2], self.fourier_weight[3])

        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = torch.fft.irfft2(out_ft, s=(N, M), norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        backcast = self.backcast_ff(x, backcast)
        forecast = self.forecast_ff(x, forecast)

        out = torch.stack([backcast, forecast], dim=1)
        # out.shape == [batch_size, 2, grid_size, grid_size, out_dim]

        out = rearrange(out, 'b k m n i -> b (k m n i) 1')

        return out


class DEQBlock(nn.Module):
    def __init__(self, modes, width, n_layers, size, pretraining):
        super().__init__()
        self.n_layers = n_layers
        self.pretraining = pretraining
        self.width = width
        self.f = SpectralConv2d(in_dim=width,
                                out_dim=width,
                                n_modes=modes,
                                size=size)
        self.solver = broyden

    def forward(self, z0, x):
        # z0.shape == [n_batches, width, flat_size]
        if self.pretraining:
            z = z0
            for _ in range(self.n_layers):
                z = self.f(z, x)
            return z

        f_thres = 24
        b_thres = 24

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
    def __init__(self, modes, width, input_dim, n_layers, size, pretraining=True):
        super().__init__()
        self.width = width
        self.in_proj = nn.Linear(input_dim, self.width)
        self.deq_block = DEQBlock(modes, width, n_layers, size, pretraining)
        self.out = nn.Linear(self.width, 1)
        self.solver = broyden
        self.gnorm = nn.GroupNorm(width // 16, width)

    def forward(self, x):
        _, N, M, _ = x.shape
        # x.shape == [n_batches, *dim_sizes, input_size]

        x = self.in_proj(x)
        x = rearrange(x, 'b m n w -> b w m n')
        x = self.gnorm(x)
        # x.shape == [n_batches, *dim_sizes, width]

        x = rearrange(x, 'b w m n -> b (m n w) 1')
        B, T, _ = x.shape
        # x.shape == [n_batches, flat_size, 1]

        z0 = x.new_zeros([B, 2 * T, 1])
        # z0.shape == [n_batches, 2 * flat_size, 1]

        new_z_star = self.deq_block(z0, x)

        new_z_star = rearrange(new_z_star, 'b (k m n w) 1 -> b k m n w',
                               k=2, n=N, m=M, w=self.width)

        forecast = self.out(new_z_star[:, 1])

        return forecast
