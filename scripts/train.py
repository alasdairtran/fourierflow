import math
import pickle
from collections import OrderedDict

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

# from torchdiffeq import odeint_adjoint as odeint


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


class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=64, nhidden=256):
        super().__init__()
        self.fc1 = GehringLinear(latent_dim, nhidden)
        self.fc2 = GehringLinear(nhidden, nhidden)
        self.fc3 = GehringLinear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = F.gelu(out)
        out = self.fc2(out)
        out = F.gelu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = GehringLinear(input_dim, hidden_dim)
        self.fc2 = GehringLinear(hidden_dim, latent_dim * 2)

    def forward(self, x):
        out = self.fc2(F.gelu(self.fc1(x)))
        return out


class Decoder(nn.Module):
    def __init__(self, latent_dim=64, obs_dim=1, nhidden=256):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = GehringLinear(latent_dim, nhidden)
        self.fc2 = GehringLinear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


class VevoDataset(Dataset):
    def __init__(self, split):
        data_path = '/localdata/u4921817/projects/phd/radflow/data/vevo/vevo_static.hdf5'
        views = h5py.File(data_path, 'r')['views'][...]

        with open('/localdata/u4921817/projects/phd/radflow/data/vevo/vevo_static_connected_nodes.pkl', 'rb') as f:
            test_nodes = sorted(pickle.load(f))

        self.samples = torch.log1p(torch.from_numpy(views[test_nodes]).float())
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
                 latent_dim=64,
                 hidden_dim=128,
                 backcast_len=42,
                 forecast_len=7,
                 obs_dim=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.func = LatentODEfunc(latent_dim, hidden_dim)
        self.rec = RecognitionRNN(backcast_len, hidden_dim, latent_dim)
        self.dec = Decoder(latent_dim, obs_dim, hidden_dim)
        total_len = backcast_len + forecast_len
        self.register_buffer('ts', torch.arange(total_len).float() / total_len)
        self.noise_std = 0.2

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def forward_ode(self, batch):
        device = batch.device
        # batch.shape == [batch_size, total_steps]

        # Transform backcast into latent space
        out = self.rec(batch[:, :-7])

        # Obtain mean and variance
        qz0_mean = out[:, :self.latent_dim]
        qz0_logvar = out[:, self.latent_dim:]

        # Sample initial state
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        # forward in time and solve ode for reconstructions
        pred_z = odeint(self.func, z0, self.ts,
                        method='dopri5').permute(1, 0, 2)
        # pred_z.shape == [batch_size, total_steps, latent_dim]

        pred_x = self.dec(pred_z)
        # pred_z.shape == [batch_size, total_steps, 1]

        pred_x = pred_x.squeeze(-1)
        # pred_z.shape == [batch_size, total_steps]

        return pred_x, z0, qz0_mean, qz0_logvar

    def training_step(self, batch, batch_idx):
        device = batch.device
        # batch.shape == [batch_size, total_steps]

        pred_x, z0, qz0_mean, qz0_logvar = self.forward_ode(batch)
        # pred_z.shape == [batch_size, total_steps]

        # Compute reconstruction loss
        noise_std_ = torch.zeros(pred_x.size()).to(device) + self.noise_std
        noise_logvar = 2. * torch.log(noise_std_).to(device)
        log_px = log_normal_pdf(batch, pred_x, noise_logvar).sum(-1).sum(-1)

        # Compute KL loss - we want the sampling distribution to be as close
        # to the standard normal as possible.
        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)
        loss = torch.mean(-log_px + analytic_kl, dim=0)

        # Compute SMAPE
        forecasts = torch.exp(pred_x[:, -7:]).detach().cpu().numpy() - 1
        targets = torch.exp(batch[:, -7:]).cpu().numpy() - 1

        denominator = np.abs(forecasts) + np.abs(targets)
        smape = 200 * np.abs(forecasts - targets) / denominator
        smape = np.nan_to_num(smape, nan=200, posinf=200, neginf=200)
        smape = smape.mean()

        self.log('train_loss', loss)
        self.log('train_smape', smape)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_x, _, _, _ = self.forward_ode(batch)
        # pred_z.shape == [batch_size, total_steps]

        # Compute SMAPE
        forecasts = torch.exp(pred_x[:, -7:]).detach().cpu().numpy() - 1
        targets = torch.exp(batch[:, -7:]).cpu().numpy() - 1

        denominator = np.abs(forecasts) + np.abs(targets)
        smape = 200 * np.abs(forecasts - targets) / denominator
        smape = np.nan_to_num(smape, nan=200, posinf=200, neginf=200)
        smape = smape.sum()

        return {'val_smape': smape, 'batch_size': len(pred_x)}

    def validation_epoch_end(self, outputs):
        val_smape_mean = 0
        count = 0

        for output in outputs:
            val_smape = output['val_smape']
            val_smape_mean += val_smape
            count += output['batch_size']
        val_smape_mean /= count
        print('test smape', val_smape_mean)
        self.log('val_smape', val_smape_mean)

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = F.nll_loss(logits, y)
    #     self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer


def main():
    train_loader = DataLoader(VevoDataset('train'),
                              batch_size=32,
                              shuffle=True,
                              num_workers=1)

    valid_loader = DataLoader(VevoDataset('valid'),
                              batch_size=32, num_workers=1)
    test_loader = DataLoader(VevoDataset('test'), batch_size=32, num_workers=1)

    ts_ode = TimeSeriesODE()
    trainer = pl.Trainer(gpus=1, max_epochs=40)
    trainer.fit(ts_ode, train_loader, test_loader)

    # result = trainer.test(ts_ode, test_loader)


if __name__ == '__main__':
    main()
