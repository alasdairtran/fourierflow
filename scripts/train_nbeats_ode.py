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

        self.samples = torch.log1p(torch.from_numpy(views).float())
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
                 forecast_length=7,
                 backcast_length=42,
                 hidden_size=128,
                 dropout=0.1):
        super().__init__()
        self.in_proj = nn.Sequential(
            GehringLinear(backcast_length, hidden_size),
            nn.ReLU(inplace=True),
            GehringLinear(hidden_size, hidden_size),
        )

        f = nn.Sequential(
            GehringLinear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            GehringLinear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            GehringLinear(hidden_size, hidden_size),
        )
        self.model = NeuralDE(f, sensitivity='autograd',
                              solver='dopri5')

        self.out_proj = nn.Sequential(
            GehringLinear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            GehringLinear(hidden_size, 1),
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # Transform backcast into latent space
        h = self.in_proj(batch[:, :-7])
        s_span = torch.linspace(0, 1, 7)
        trajectory = self.model.trajectory(h, s_span).transpose(0, 1)
        # trajectory.shape == [batch_size, forecast_len, hidden_size]
        forecasts = self.out_proj(trajectory).squeeze(-1)
        forecasts = torch.exp(forecasts) - 1
        targets = torch.exp(batch[:, -7:]) - 1

        denominator = torch.abs(forecasts) + torch.abs(targets)
        smape = 200 * torch.abs(forecasts - targets) / denominator
        # smape = torch.nan_to_num(smape, nan=200, posinf=200, neginf=200)
        smape[torch.isnan(smape)] = 0
        smape = smape.mean()
        self.log('train_smape', smape)

        return smape

    def validation_step(self, batch, batch_idx):
        # Transform backcast into latent space
        h = self.in_proj(batch[:, :-7])
        s_span = torch.linspace(0, 1, 7)
        trajectory = self.model.trajectory(h, s_span).transpose(0, 1)
        # trajectory.shape == [batch_size, forecast_len, hidden_size]
        forecasts = self.out_proj(trajectory).squeeze(-1)
        forecasts = torch.exp(forecasts) - 1
        targets = torch.exp(batch[:, -7:]) - 1

        denominator = torch.abs(forecasts) + torch.abs(targets)
        smape = 200 * torch.abs(forecasts - targets) / denominator
        # smape = torch.nan_to_num(smape, nan=200, posinf=200, neginf=200)
        smape = smape.sum()

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
        num_training_steps = 950 * 40  # 215 * 40
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
