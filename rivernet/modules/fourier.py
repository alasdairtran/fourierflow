from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes

        self.fourier_weight = nn.Parameter(torch.FloatTensor(
            in_dim, out_dim, n_modes, 2))
        nn.init.xavier_normal_(self.fourier_weight, gain=1/(in_dim*out_dim))

    @staticmethod
    def complex_matmul_1d(a, b):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        op = partial(torch.einsum, "bix,iox->box")

        # Recall multiplication of two complex numbers:
        # (x + yi)(u + vi) = (xu - yv) + (xv + yu)i
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):
        # x.shape == [batch_size, in_dim, n_steps]
        B, I, N = x.shape

        # Fourier transform in the space dimension
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # x_ft's final dimension represent complex coefficients
        # The Fourier modes are sorted from lowest frequency to highest.
        x_ft = torch.fft.rfft(x, n=N, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, n_steps // 2 + 1]

        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=3)
        # x_ft.shape == [batch_size, in_dim, n_steps // 2 + 1, 2]

        # Step 1: Zero out all modes less than top k
        out_ft = torch.zeros(B, I, N // 2 + 1, 2, device=x.device)
        # out_ft.shape == [batch_size, in_dim, n_steps // 2 + 1, 2]

        # Multiply relevant Fourier modes
        out_ft[:, :, :self.n_modes] = self.complex_matmul_1d(
            x_ft[:, :, :self.n_modes], self.fourier_weight)

        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')

        return x


class SimpleBlock1d(nn.Module):
    def __init__(self, modes, width):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
