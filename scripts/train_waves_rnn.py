import math
import os
import pickle

import comet_ml
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import SineData
from models import NeuralODEProcess
from pytorch_lightning.loggers import CometLogger
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.data import DataLoader, Dataset
from torchdiffeq import odeint
from torchdyn.models import NeuralDE
from viz import plot_rnn_sines


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


class TimeSeriesODE(pl.LightningModule):
    def __init__(self,
                 x_dim=1,
                 y_dim=1,
                 r_dim=50,
                 z_dim=50,
                 h_dim=50,
                 L_dim=10,
                 initial_x=-3.2,
                 forecast_length=7,
                 backcast_length=42,
                 hidden_size=256,
                 latent_size=256,
                 l_size=64,
                 dropout=0.1,
                 num_context_range=(1, 10),
                 extra_target_range=(0, 5),
                 testing_context_size=10):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.l_size = l_size
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.nfe = 0
        self.num_context_range = num_context_range
        self.num_extra_target_range = extra_target_range
        self.test_context_size = testing_context_size
        self.rs = np.random.RandomState(1242)

        initial_x = torch.FloatTensor([initial_x]).view(1, 1, 1)
        self.rnn = nn.LSTM(1, 32, 1, batch_first=True, dropout=0.2,
                           bidirectional=True)
        self.out_proj = GehringLinear(32 * 2, 1)

        # Fix a random datapoint for plotting
        dataset = SineData(amplitude_range=(0.7, 0.9),
                           shift_range=(0.0, 0.1),
                           num_samples=1)
        self.t = dataset[0][0]
        self.y = dataset[0][1]
        self.mu = dataset[0][2]

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y, mu = batch

        # Sample number of context and target points
        num_context = self.rs.randint(*self.num_context_range)
        num_extra_target = self.rs.randint(*self.num_extra_target_range)

        y_context, y_target, mask = context_target_split(
            x, y, num_context, num_extra_target)

        outputs, _ = self.rnn(y_context)
        # outputs.shape == [batch_size, seq_len, hidden_size * 2]

        preds = self.out_proj(outputs)
        loss = F.mse_loss(y_target[mask], preds[mask])
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mu = batch

        # Sample number of context and target points
        num_context = self.rs.randint(*self.num_context_range)
        num_extra_target = self.rs.randint(*self.num_extra_target_range)

        y_context, y_target, mask = context_target_split(
            x, y, num_context, num_extra_target)

        outputs, _ = self.rnn(y_context)
        # outputs.shape == [batch_size, seq_len, hidden_size * 2]

        preds = self.out_proj(outputs)
        loss = F.mse_loss(y_target[mask], preds[mask])
        self.log('valid_loss', loss)

    def validation_epoch_end(self, outputs):
        device = next(self.parameters()).device
        self.training = True
        plot_rnn_sines(device, self.t, self.y, self.mu,
                       self.rnn, self.out_proj, self.logger.experiment)
        self.training = False

    def configure_optimizers(self):
        opt = torch.optim.RMSprop(self.parameters(), lr=1e-3)
        return opt


def context_target_split(x, y, num_context, num_extra_target, locations=None):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.

    Parameters
    ----------
    x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

    y : torch.Tensor
            Shape (batch_size, num_points, y_dim)

    num_context : int
            Number of context points.

    num_extra_target : int
            Number of additional target points.
    """
    num_points = x.shape[1]
    # Sample locations of context and target points
    if locations is None:
        locations = np.random.choice(num_points,
                                     size=num_context + num_extra_target,
                                     replace=False)

    y_context = y.new_full(y.shape, -999)
    y_context[:, locations[:num_context]] = y[:, locations[:num_context]]

    y_target = y.new_full(y.shape, -999)
    y_target[:, locations] = y[:, locations]

    # B, S, _ = y_context.shape
    # y_context = y_context.reshape(B, S)
    # mask = (y_context == -999).cpu().numpy()
    # idx = np.where(~mask, torch.arange(mask.shape[1]), 0)
    # np.maximum.accumulate(idx, axis=1, out=idx)
    # y_context = y_context[torch.arange(idx.shape[0])[:, None], idx]
    # y_context = y_context.reshape(B, S, 1)
    y_context[y_context == -999] = 0

    mask = y_target != -999

    return y_context, y_target, mask


def main():
    test_set_size = 10
    batch_size = 5
    dataset = SineData(amplitude_range=(-1., 1.),
                       shift_range=(-.5, .5),
                       num_samples=500)
    train_loader = DataLoader(dataset[:int(len(dataset)-test_set_size)],
                              batch_size=batch_size, shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(dataset[int(len(dataset)-test_set_size):],
                             batch_size=test_set_size, shuffle=False,
                             num_workers=4)

    comet_logger = CometLogger(
        api_key=os.environ.get('COMET_API_KEY'),
        workspace=os.environ.get('COMET_WORKSPACE'),
        save_dir='expt',
        project_name='neural-ode',
        rest_api_key=os.environ.get('COMET_REST_API_KEY'),
        experiment_name=os.environ.get('COMET_EXPERIMENT_KEY', 'rnn_1')
    )

    ts_ode = TimeSeriesODE()
    trainer = pl.Trainer(gpus=1, max_epochs=40, gradient_clip_val=0.1,
                         logger=comet_logger)
    trainer.fit(ts_ode, train_loader, test_loader)

    # result = trainer.test(ts_ode, test_loader)


if __name__ == '__main__':
    main()
