# Source: https://github.com/philipperemy/n-beats/blob/master/nbeats_pytorch/model.py
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from torch.nn import functional as F

from rivernet.modules.linear import GehringLinear

from .base import Module
from .fourier import SpectralConv1d


@Module.register('nbeats')
class NBeatsNet(Module):
    def __init__(self,
                 n_stacks=4,
                 nb_blocks_per_stack=3,
                 forecast_length=5,
                 backcast_length=10,
                 share_weights_in_stack=False,
                 hidden_layer_units=256,
                 nb_harmonics=None,
                 block_type='generic',
                 dropout=0.1):
        super().__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stacks = nn.ModuleList([])
        self.dropout = dropout
        self.block_type = block_type
        for stack_id in range(n_stacks):
            self.stacks.append(self.create_stack(stack_id))

    def create_stack(self, stack_id):
        blocks = nn.ModuleList([])
        for block_id in range(self.nb_blocks_per_stack):
            if self.block_type == 'generic':
                block_init = GenericBlock
            elif self.block_type == 'fourier':
                block_init = FourierBlock
            if self.share_weights_in_stack and block_id != 0:
                # pick up the last one when we share weights.
                block = blocks[-1]
            else:
                block = block_init(self.hidden_layer_units, self.hidden_layer_units,
                                   self.backcast_length, self.forecast_length,
                                   self.nb_harmonics, self.dropout)
            blocks.append(block)
        return blocks

    def forward(self, backcast):
        B = backcast.shape[0]
        forecast = torch.zeros(size=(B, self.forecast_length))
        forecast = forecast.to(backcast.device)

        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)

                backcast = backcast - b
                # backcast.shape == [B, S]

                forecast = forecast + f
                # forecast.shape == [B, T]

        return backcast, forecast


def linspace(backcast_length, forecast_length):
    lin_space = np.linspace(-backcast_length, forecast_length,
                            backcast_length + forecast_length)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class FourierBlock(nn.Module):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, share_thetas=False,
                 nb_harmonics=None, dropout=0.0):
        super().__init__()
        self.theta_f_fc = self.theta_b_fc = GehringLinear(
            units, thetas_dim, bias=False)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = GehringLinear(
                units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = GehringLinear(
                backcast_length, thetas_dim, bias=False)
            self.theta_f_fc = GehringLinear(
                backcast_length, thetas_dim, bias=False)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))
        self.register_buffer('_float', torch.FloatTensor(1))

        width = units
        n_modes = 8
        self.fc0 = nn.Linear(2, width)

        self.conv0 = SpectralConv1d(width, width, n_modes)
        # self.conv1 = SpectralConv1d(width, width, n_modes)
        # self.conv2 = SpectralConv1d(width, width, n_modes)
        # self.conv3 = SpectralConv1d(width, width, n_modes)
        self.w0 = nn.Conv1d(width, width, 1)
        # self.w1 = nn.Conv1d(width, width, 1)
        # self.w2 = nn.Conv1d(width, width, 1)
        # self.w3 = nn.Conv1d(width, width, 1)

        self.fc1 = nn.Linear(width, 80)
        self.fc2 = nn.Linear(80, 1)

        self.backcast_fc = GehringLinear(thetas_dim, backcast_length)
        self.forecast_fc = GehringLinear(thetas_dim, forecast_length)

    def forward(self, x):
        # x.shape == [batch_size, backcast_len]

        x = x.unsqueeze(-1)
        # x.shape == [batch_size, backcast_len, 1]

        # Add positional information
        pos = torch.linspace(0, 1, x.shape[1]).to(x.device)
        # pos.shape == [backcast_len]

        pos = repeat(pos, 's -> b s d', b=x.shape[0], d=1)
        # pos.shape == [batch_size, backcast_len, 1]

        x = torch.cat([x, pos], dim=-1)
        # x.shape == [batch_size, backcast_len, 2]

        x = self.fc0(x)
        x = rearrange(x, 'b s w -> b w s')
        # x.shape == [batch_size, width, backcast_len]

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        # x = self.conv1(x)
        # x2 = self.w1(x)
        # x = x + x1
        # x = F.relu(x)

        # x = self.conv2(x)
        # x2 = self.w2(x)
        # x = x + x1
        # x = F.relu(x)

        # x = self.conv3(x)
        # x2 = self.w3(x)
        # x = x + x1

        x = rearrange(x, 'b w s-> b s w')

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.squeeze(-1)
        # x.shape == [batch_size, backcast_len]

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast


class Block(nn.Module):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, share_thetas=False,
                 nb_harmonics=None, dropout=0.0):
        super().__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = GehringLinear(backcast_length, units, dropout=dropout)
        self.fc2 = GehringLinear(units, units, dropout=dropout)
        self.fc3 = GehringLinear(units, units, dropout=dropout)
        self.fc4 = GehringLinear(units, units, dropout=dropout)
        self.dropout = dropout
        self.backcast_linspace, self.forecast_linspace = linspace(
            backcast_length, forecast_length)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = GehringLinear(
                units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = GehringLinear(units, thetas_dim, bias=False)
            self.theta_f_fc = GehringLinear(units, thetas_dim, bias=False)

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))
        self.register_buffer('_float', torch.FloatTensor(1))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, self.dropout, self.training)

        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, nb_harmonics=None, dropout=0.1):
        super().__init__(units, thetas_dim, backcast_length, forecast_length,
                         dropout=dropout)

        self.backcast_fc = GehringLinear(thetas_dim, backcast_length)
        self.forecast_fc = GehringLinear(thetas_dim, forecast_length)

    def forward(self, x):
        # no constraint for generic arch.
        x = super().forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast
