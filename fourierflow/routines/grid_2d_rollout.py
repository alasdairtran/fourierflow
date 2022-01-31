from typing import Any, Dict

import torch
import torch.nn as nn
import wandb
from einops import rearrange, repeat

from fourierflow.modules import fourier_encode
from fourierflow.modules.loss import LpLoss

from .base import Routine


class Grid2DRolloutExperiment(Routine):
    def __init__(self,
                 conv: nn.Module,
                 n_steps: int,
                 k_max: int = 32,
                 num_freq_bands: int = 8,
                 freq_base: int = 2,
                 use_fourier_position: bool = False,
                 append_pos: bool = True,
                 teacher_forcing: bool = False,
                 step_size: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.conv = conv
        self.n_steps = n_steps
        self.l2_loss = LpLoss(size_average=True)
        self.use_fourier_position = use_fourier_position
        self.k_max = k_max
        self.num_freq_bands = num_freq_bands
        self.freq_base = freq_base
        self.append_pos = append_pos
        self.teacher_forcing = teacher_forcing
        self.step_size = step_size
        if self.use_fourier_position:
            self.in_proj = nn.Linear(n_steps, 34)

    def forward(self, data):
        xx = data[..., :10]
        B, X, Y, T = xx.shape

        # Add positional information to inputs
        ticks = torch.linspace(0, 1, X).to(xx.device)
        grid_x = repeat(ticks, 'x -> b x y 1', b=B, y=Y)
        grid_y = repeat(ticks, 'y -> b x y 1', b=B, x=X)
        xx = torch.cat([xx, grid_x, grid_y], dim=-1)

        yy = data[..., 10:]
        return self._learning_step([xx, yy])

    def encode_fourier_positions(self, dim_sizes, device):
        # dim_sizes is a list of dimensions in all positional dimensions
        # e.g. for a 64 x 64 image, dim_sizes = [64, 64]

        # A way to interpret `pos` is that we could append `pos` directly
        # to the raw inputs to attach the positional info to the raw features.
        def generate_grid(size):
            return torch.linspace(-1., 1., steps=size, device=device)
        grid_list = list(map(generate_grid, dim_sizes))
        pos = torch.stack(torch.meshgrid(*grid_list, indexing='ij'), dim=-1)
        # pos.shape == [*dim_sizes, n_dims]

        # To get the fourier encodings, we will go one step further
        fourier_feats = fourier_encode(
            pos, self.k_max, self.num_freq_bands, base=self.freq_base)
        # fourier_feats.shape == [*dim_sizes, n_dims, n_bands * 2 + 1]

        fourier_feats = rearrange(fourier_feats, '... n d -> ... (n d)')
        # fourier_feats.shape == [*dim_sizes, pos_size]

        return fourier_feats

    def _learning_step(self, batch):
        xx, yy = batch['x'], batch['y']
        B, *dim_sizes, _ = xx.shape
        X, Y = dim_sizes
        # Note here the in_channels consists of the first 10 time steps
        # and 2 positional encodings.
        # xx.shape == [batch_size, x_dim, y_dim, in_channels]
        # yy.shape == [batch_size, x_dim, y_dim, out_channels]

        if self.use_fourier_position:
            pos_feats = self.encode_fourier_positions(dim_sizes, xx.device)
            # pos_feats.shape == [*dim_sizes, pos_size]

            pos_feats = repeat(pos_feats, '... -> b ...', b=B)
            # pos_feats.shape == [batch_size, *dim_sizes, pos_size]

            # xx = torch.cat([xx[..., :-2], pos_feats], dim=-1)
            # xx.shape == [batch_size, x_dim, y_dim, in_channels]

            xx = xx[..., :-2]
            embeds = self.in_proj(xx) + pos_feats

            P = pos_feats.shape[-1]

        else:
            ticks = torch.linspace(0, 1, X).to(xx.device)
            grid_x = repeat(ticks, 'x -> b x y 1', b=B, y=Y)
            grid_y = repeat(ticks, 'y -> b x y 1', b=B, x=X)
            # grid_x.shape == [batch_size, *dim_sizes, 1]

            pos_feats = torch.cat([grid_x, grid_y], dim=-1)
            # pos_feats.shape == [batch_size, *dim_sizes, 2]

            embeds = xx
            P = 2

        loss = 0
        step_losses = []
        # We predict one future one step at a time
        for t in range(self.n_steps):
            y = yy[..., t: t+1]
            # y.shape == [batch_size, x_dim, y_dim, 1]

            out = self.conv(embeds)
            im = out['forecast']
            # im.shape == [batch_size, *dim_sizes, 1]

            l = self.l2_loss(im.reshape(B, -1), y.reshape(B, -1))
            step_losses.append(l)
            loss += l
            pred = im if t == 0 else torch.cat((pred, im), dim=-1)

            if self.teacher_forcing and self.training:
                im = y

            if self.use_fourier_position:
                xx = torch.cat((xx[..., 1:], im), dim=-1)
                embeds = self.in_proj(xx) + pos_feats
            elif self.append_pos:
                embeds = torch.cat((embeds[..., 1: -P], im, pos_feats), dim=-1)
            else:
                embeds = torch.cat((embeds[..., 1:], im), dim=-1)

        loss /= self.n_steps
        loss_full = self.l2_loss(pred.reshape(B, -1), yy.reshape(B, -1))

        pred_norm = torch.norm(pred, dim=[1, 2], keepdim=True)
        yy_norm = torch.norm(yy, dim=[1, 2], keepdim=True)
        p = (pred / pred_norm) * (yy / yy_norm)
        p = p.sum(dim=[1, 2]).mean(dim=0)
        # p.shape == [n_steps]

        has_diverged = p < 0.95
        diverged_idx = has_diverged.nonzero()
        diverged_t = diverged_idx[0, 0] if len(
            diverged_idx) > 0 else len(has_diverged)
        time_until = diverged_t * self.step_size

        return loss, loss_full, pred, step_losses, p, time_until

    def training_step(self, batch, batch_idx):
        loss, loss_full, _, _, _, _ = self._learning_step(batch)
        self.log('train_loss', loss)
        self.log('train_loss_full', loss_full)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_full, preds, _, _, time_until = self._learning_step(batch)
        self.log('valid_loss_avg', loss)
        self.log('valid_loss', loss_full, prog_bar=True)
        self.log('valid_time_until', time_until, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, loss_full, _, step_losses, p, time_until = self._learning_step(
            batch)
        self.log('test_loss_avg', loss)
        self.log('test_loss', loss_full)
        self.log('test_time_until', time_until)

        if self.logger:
            times = batch['times'].cpu().numpy()
            corr_rows = list(zip(times, p.cpu().numpy()))
            self.logger.experiment.log({
                'test_correlations': wandb.Table(['time', 'corr'], corr_rows)})

            loss_rows = list(zip(times, step_losses))
            self.logger.experiment.log({
                'test_losses': wandb.Table(['time', 'loss'], loss_rows)})
