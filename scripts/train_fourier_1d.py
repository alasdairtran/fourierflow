import operator
from functools import partial, reduce

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from util import LpLoss, MatReader


def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    op = partial(torch.einsum, "bix,iox->box")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 1, normalized=True, onesided=True)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels,
                             x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1] = compl_mul1d(
            x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.irfft(out_ft, 1, normalized=True,
                        onesided=True, signal_sizes=(x.size(-1), ))
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


class Net1D(pl.LightningModule):
    def __init__(self, modes, width):
        super().__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock1d(modes, width)
        self.myloss = LpLoss(size_average=False)

    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        B = x.shape[0]
        out = self.forward(x)
        mse = F.mse_loss(out, y, reduction='mean')
        l2 = self.myloss(out.view(B, -1), y.view(B, -1))
        self.log('train_mse', mse)
        self.log('train_loss', l2)
        return l2

    def validation_step(self, batch, batch_idx):
        x, y = batch
        B = x.shape[0]
        out = self.forward(x)
        l2 = self.myloss(out.view(B, -1), y.view(B, -1))
        self.log('valid_loss', l2 / B)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
        return [opt], [sched]


def main():
    ntrain = 1000
    ntest = 100
    sub = 2**3  # subsampling rate
    h = 2**13 // sub  # total grid size divided by the subsampling rate
    s = h
    batch_size = 20
    modes = 16
    width = 64

    dataloader = MatReader('data/burgers_data_R10.mat')
    x_data = dataloader.read_field('a')[:, ::sub]
    y_data = dataloader.read_field('u')[:, ::sub]

    x_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :]
    x_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :]

    # cat the locations information
    grid = np.linspace(0, 2*np.pi, s).reshape(1, s, 1)
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat([x_train.reshape(ntrain, s, 1),
                         grid.repeat(ntrain, 1, 1)], dim=2)
    x_test = torch.cat([x_test.reshape(ntest, s, 1),
                        grid.repeat(ntest, 1, 1)], dim=2)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        x_test, y_test), batch_size=batch_size, shuffle=False)

    ts_ode = Net1D(modes, width)
    trainer = pl.Trainer(gpus=1, max_epochs=500)
    trainer.fit(ts_ode, train_loader, test_loader)

    # result = trainer.test(ts_ode, test_loader)


if __name__ == '__main__':
    main()
