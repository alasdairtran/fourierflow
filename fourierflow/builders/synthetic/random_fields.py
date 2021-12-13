"""Generate random Gaussian fields.

Adapted from:
https://github.com/zongyi-li/fourier_neural_operator/blob/master/data_generation/navier_stokes/random_fields.py
"""
import math

import torch


class GaussianRF:
    def __init__(self, n_dims, size, alpha=2, tau=3, sigma=None, device=None):

        self.n_dims = n_dims
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.n_dims))

        k_max = size//2

        if n_dims == 1:
            self.dim = [-1]
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size * \
                math.sqrt(2.0)*sigma * \
                ((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif n_dims == 2:
            self.dim = [-1, -2]
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size, 1)

            k_x = wavenumers.transpose(0, 1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma * \
                ((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0, 0] = 0.0

        elif n_dims == 3:
            self.dim = [-1, -2, -3]
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size, size, 1)

            k_x = wavenumers.transpose(1, 2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0, 2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)
                                                             * (k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0, 0, 0] = 0.0

        self.size = []
        for _ in range(self.n_dims):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):
        coeff = torch.randn(N, *self.size, 2, device=self.device)

        coeff[..., 0] = self.sqrt_eig*coeff[..., 0]
        coeff[..., 1] = self.sqrt_eig*coeff[..., 1]

        u = torch.fft.ifftn(torch.view_as_complex(
            coeff), dim=self.dim, norm='backward').real

        return u
