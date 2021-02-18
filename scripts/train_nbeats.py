# Source: https://github.com/philipperemy/n-beats/blob/master/nbeats_pytorch/model.py
import math
import pickle

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchdyn.models import NeuralDE


class GehringLinear(nn.Linear):
    """A linear layer with Gehring initialization and weight normalization."""

    def __init__(self, in_features, out_features, dropout=0, bias=True,
                 weight_norm=True):
        self.dropout = dropout
        self.weight_norm = weight_norm
        super().__init__(in_features, out_features, bias)

    def reset_parameters(self):
        std = math.sqrt((1 - self.dropout) / self.in_features)
        self.weight.data.normal_(mean=0, std=std)
        if self.bias is not None:
            self.bias.data.fill_(0)

        if self.weight_norm:
            nn.utils.weight_norm(self)


class NBeatsNet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self,
                 device,
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=3,
                 forecast_length=5,
                 backcast_length=10,
                 thetas_dims=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=256,
                 nb_harmonics=None,
                 dropout=0.1):
        super().__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = nn.ModuleList([])
        self.thetas_dim = thetas_dims
        self.device = device
        self.dropout = dropout
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.to(self.device)

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        blocks = nn.ModuleList([])
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                # pick up the last one when we share weights.
                block = blocks[-1]
            else:
                block = block_init(self.hidden_layer_units, self.thetas_dim[stack_id],
                                   self.device, self.backcast_length, self.forecast_length,
                                   self.nb_harmonics, self.dropout)
            blocks.append(block)
        return blocks

    @staticmethod
    def select_block(block_type):
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def forward(self, backcast):
        # maybe batch size here.
        B = backcast.shape[0]
        backcast = backcast.to(self.device)

        forecast = torch.zeros(size=(B, self.forecast_length)).to(self.device)

        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)

                backcast = backcast - b
                # backcast.shape == [B, S]

                forecast = forecast + f
                # forecast.shape == [B, T]

        return forecast


def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor([np.cos(2 * np.pi * i * t)
                       for i in range(p1)]).float()  # H/2-1
    s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(device))


def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.tensor([t ** i for i in range(p)]).float()
    return thetas.mm(T.to(device))


def linspace(backcast_length, forecast_length):
    lin_space = np.linspace(-backcast_length, forecast_length,
                            backcast_length + forecast_length)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class Block(nn.Module):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, share_thetas=False,
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
        self.device = device
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
        x = F.relu(self.fc1(x.to(self.device)))
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


class SeasonalityBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None, dropout=0.1):
        if nb_harmonics:
            super().__init__(units, nb_harmonics, device, backcast_length,
                             forecast_length, share_thetas=True,
                             dropout=dropout)
        else:
            super().__init__(units, forecast_length, device, backcast_length,
                             forecast_length, share_thetas=True,
                             dropout=dropout)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self.theta_b_fc(
            x), self.backcast_linspace, self.device)
        forecast = seasonality_model(self.theta_f_fc(
            x), self.forecast_linspace, self.device)
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None, dropout=0.1):
        super(TrendBlock, self).__init__(units, thetas_dim, device, backcast_length,
                                         forecast_length, share_thetas=True,
                                         dropout=dropout)

    def forward(self, x):
        x = super().forward(x)
        backcast = trend_model(self.theta_b_fc(
            x), self.backcast_linspace, self.device)
        forecast = trend_model(self.theta_f_fc(
            x), self.forecast_linspace, self.device)
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None, dropout=0.1):
        super().__init__(units, thetas_dim,
                         device, backcast_length, forecast_length,
                         dropout=dropout)

        self.backcast_fc = GehringLinear(thetas_dim, backcast_length)
        self.forecast_fc = GehringLinear(thetas_dim, forecast_length)

        # if max_neighbours > 0:
        #     self.backcast_fc_n = GehringLinear(thetas_dim, backcast_length)
        #     self.forecast_fc_n = GehringLinear(thetas_dim, forecast_length)

    def forward(self, x):
        # no constraint for generic arch.
        x = super().forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast


class VevoDataset(Dataset):
    def __init__(self, split):
        data_path = '/localdata/u4921817/projects/phd/radflow/data/vevo/vevo_static.hdf5'
        views = h5py.File(data_path, 'r')['views'][...]

        # if split == 'train':
        #     with open('/localdata/u4921817/projects/phd/radflow/data/vevo/vevo_all_nodes.pkl', 'rb') as f:
        #         indices = sorted(pickle.load(f))
        #         views = views[indices]
        # else:
        if True:
            with open('/localdata/u4921817/projects/phd/radflow/data/vevo/vevo_static_connected_nodes.pkl', 'rb') as f:
                indices = sorted(pickle.load(f))
                views = views[indices]

        # Forward-fill missing values
        mask = views == -1
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        views = views[np.arange(idx.shape[0])[:, None], idx]

        # Fill remaining missing values with 0
        views[views == -1] = 0

        assert (views >= 0).all()

        self.samples = torch.log1p(torch.from_numpy(views).float())
        assert not self.samples.isnan().any()
        if split == 'train':
            self.samples = self.samples[:, :49]
        elif split == 'valid':
            self.samples = self.samples[:, 7:56]
        elif split == 'test':
            self.samples = self.samples[:, 14:63]
        else:
            raise ValueError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TimeSeriesODE(pl.LightningModule):
    def __init__(self,
                 n_stacks=4,
                 nb_blocks_per_stack=1,
                 forecast_length=7,
                 backcast_length=42,
                 thetas_dims=128,
                 hidden_size=128,
                 share_weights_in_stack=False,
                 dropout=0.1):
        super().__init__()
        # self.model = NBeatsNet(device=torch.device('cuda:0'),
        #                        stack_types=[
        #                            NBeatsNet.GENERIC_BLOCK] * n_stacks,
        #                        nb_blocks_per_stack=nb_blocks_per_stack,
        #                        forecast_length=forecast_length,
        #                        backcast_length=backcast_length,
        #                        thetas_dims=[thetas_dims] * n_stacks,
        #                        hidden_layer_units=hidden_size,
        #                        share_weights_in_stack=share_weights_in_stack,
        #                        dropout=dropout)

        self.model = nn.Sequential(
            GehringLinear(backcast_length, hidden_size),
            GehringLinear(hidden_size, hidden_size, dropout=dropout),
            nn.GELU(),
            nn.Dropout(dropout),
            GehringLinear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            GehringLinear(hidden_size, forecast_length),
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # Transform backcast into latent space
        assert not batch.isnan().any()
        forecasts = self.model(batch[:, :-7])
        forecasts = torch.exp(forecasts)
        targets = torch.exp(batch[:, -7:]) - 1

        denominator = torch.abs(forecasts) + torch.abs(targets)
        smape = 200 * torch.abs(forecasts - targets) / denominator
        # smape = torch.nan_to_num(smape, nan=200, posinf=200, neginf=200)
        smape = smape.clamp(0, 200)
        smape[torch.isnan(smape)] = 0
        smape = smape.mean()
        self.log('train_smape', smape)

        return smape

    def validation_step(self, batch, batch_idx):
        # Transform backcast into latent space
        forecasts = self.model(batch[:, :-7])
        forecasts = torch.exp(forecasts) - 1
        targets = torch.exp(batch[:, -7:]) - 1

        # Sanity check: Copy Baseline should give SMAPE of 13.955
        # forecasts = torch.exp(batch[:, -8:-7].expand(-1, 7)) - 1

        denominator = torch.abs(forecasts) + torch.abs(targets)
        smape = 200 * torch.abs(forecasts - targets) / denominator
        smape = smape.clamp(0, 200)
        # smape = torch.nan_to_num(smape, nan=200, posinf=200, neginf=200)
        smape = smape[~torch.isnan(smape)].sum()

        return {'val_smape': smape, 'batch_size': len(batch)}

    def validation_epoch_end(self, outputs):
        val_smape_mean = 0
        count = 0

        for output in outputs:
            val_smape = output['val_smape']
            val_smape_mean += val_smape
            count += output['batch_size']
        val_smape_mean /= (count * 7)
        print('test smape', val_smape_mean)
        self.log('val_smape', val_smape_mean)

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = F.nll_loss(logits, y)
    #     self.log('test_loss', loss)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-3)

        num_warmup_steps = 500
        num_training_steps = 215 * 40
        last_epoch = -1
        num_cycles = 0.5

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / \
                float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        sched = {'scheduler': torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda, last_epoch),
                 'monitor': 'loss',
                 'interval': 'step',
                 'frequency': 10}
        return [opt], [sched]


def main():
    train_loader = DataLoader(VevoDataset('train'),
                              batch_size=64,
                              shuffle=True,
                              num_workers=4)

    valid_loader = DataLoader(VevoDataset('valid'),
                              batch_size=64, num_workers=4)
    test_loader = DataLoader(VevoDataset('test'), batch_size=64, num_workers=4)

    ts_ode = TimeSeriesODE()
    trainer = pl.Trainer(gpus=1, max_epochs=40, gradient_clip_val=0.1)
    trainer.fit(ts_ode, train_loader, test_loader)

    # result = trainer.test(ts_ode, test_loader)


if __name__ == '__main__':
    main()
