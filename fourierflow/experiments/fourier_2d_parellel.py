from typing import Any, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from allennlp.common import Lazy
from allennlp.training.optimizers import Optimizer
from einops import rearrange, repeat
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fourierflow.common import Experiment, Module, Scheduler
from fourierflow.modules import fourier_encode
from fourierflow.modules.loss import LpLoss
from fourierflow.modules.perceiver import (Attention, FeedForward, PreNorm,
                                           cache_fn)


@Experiment.register('fourier_2d_parellel')
class Fourier2DParallelExperiment(Experiment):
    def __init__(self,
                 conv: Module,
                 optimizer: Lazy[Optimizer],
                 n_steps: int,
                 max_freq: int = 32,
                 num_freq_bands: int = 8,
                 freq_base: int = 2,
                 scheduler: Lazy[Scheduler] = None,
                 scheduler_config: Dict[str, Any] = None,
                 use_fourier_position: bool = False,
                 append_pos: bool = True,
                 num_latents: int = 64,
                 weight_tie_layers: bool = False,
                 latent_dim: int = 128,
                 ff_dropout: float = 0.1,
                 input_dim: int = 64,
                 cross_heads: int = 1,
                 latent_heads: int = 1,
                 latent_dim_head: int = 64,
                 attn_dropout: float = 0.0,
                 n_layers: int = 6,
                 model_path: str = None):
        super().__init__()
        self.conv = conv
        self.n_steps = n_steps
        self.l2_loss = LpLoss(size_average=True)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config
        self.use_fourier_position = use_fourier_position
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.freq_base = freq_base
        self.append_pos = append_pos
        self.embed_obs = nn.Linear(1, 34)
        self.embed_inputs = nn.Linear(34, 64)

        self.latents = nn.Parameter(torch.randn(64, 64, latent_dim))

        def get_cross_attn(): return PreNorm(latent_dim, Attention(latent_dim, input_dim,
                                                                   heads=cross_heads, dim_head=cross_dim_head, dropout=attn_dropout), context_dim=input_dim)

        def get_cross_ff(): return PreNorm(
            latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        def get_latent_attn(): return PreNorm(latent_dim, Attention(
            latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout))
        def get_latent_ff(): return PreNorm(
            latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(
            cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                PreNorm(
                    latent_dim, FeedForward(latent_dim, dropout=ff_dropout)),
                PreNorm(
                    latent_dim, FeedForward(latent_dim, dropout=ff_dropout)),
            ]))

    def forward(self, x):
        x = self.conv(x)
        return x.squeeze()

    def encode_fourier_positions(self, dim_sizes, device):
        # dim_sizes is a list of dimensions in all positional/time dimensions
        # e.g. for a 64 x 64 image over 20 steps, dim_sizes = [64, 64, 20]

        # A way to interpret `pos` is that we could append `pos` directly
        # to the raw inputs to attach the positional info to the raw features.
        def generate_grid(size):
            return torch.linspace(-1., 1., steps=size, device=device)
        grid_list = list(map(generate_grid, dim_sizes))
        pos = torch.stack(torch.meshgrid(*grid_list), dim=-1)
        # pos.shape == [*dim_sizes, n_dims]

        # To get the fourier encodings, we will go one step further
        fourier_feats = fourier_encode(
            pos, self.max_freq, self.num_freq_bands, base=self.freq_base)
        # fourier_feats.shape == [*dim_sizes, n_dims, n_bands * 2 + 1]

        fourier_feats = rearrange(fourier_feats, '... n d -> ... (n d)')
        # fourier_feats.shape == [*dim_sizes, pos_size]

        return fourier_feats

    def _learning_step(self, batch):
        B, *dim_sizes, = batch.shape
        X, Y, T = dim_sizes
        # batch.shape == [batch_size, x_dim, y_dim, total_steps]

        batch = repeat(batch, '... -> ... 1')
        # batch.shape == [batch_size, x_dim, y_dim, total_steps, 1]

        pos_feats = self.encode_fourier_positions(dim_sizes, batch.device)
        # pos_feats.shape == [*dim_sizes, pos_size]

        pos_feats = repeat(pos_feats, '... p -> b ... t p', b=B, t=T)
        # pos_feats.shape == [batch_size, *dim_sizes, total_steps, pos_size]

        embeds = self.embed_obs(batch) + pos_feats
        # embeds.shape == [batch_size, x_dim, y_dim, total_steps, 34?]

        embeds = self.embed_inputs(batch)
        # embeds.shape == [batch_size, x_dim, y_dim, total_steps, 64]

        latents = repeat(self.latents, 'n d -> b n d', b=B)

        out = latents.new_zeros(*latents.shape)
        for cross_attn, backcast_f, forecast_f in self.layers:
            x = cross_attn(x, context=embeds) + x
            # x = cross_ff(x) + x
            # x.shape == [batch_size, n_latents, latent_dim]

            b = backcast_f(x)
            f = forecast_f(x)

            x = x - b
            out = out + f

        out = out / len(self.layers)
        # out.shape == [batch_size, n_latents, latent_dim]
        ####

        loss = 0
        # We predict one future one step at a time
        pred_layer_list = []
        for t in range(self.n_steps):
            y = yy[..., t: t+1]
            # y.shape == [batch_size, x_dim, y_dim, 1]

            im, im_list = self.conv(embeds)
            pred_layer_list.append(im_list)
            # im.shape == [batch_size, *dim_sizes, 1]

            loss += self.l2_loss(im.reshape(B, -1), y.reshape(B, -1))
            pred = im if t == 0 else torch.cat((pred, im), dim=-1)
            if self.use_fourier_position:
                xx = torch.cat((xx[..., 1:], im), dim=-1)
                embeds = self.in_proj(xx) + pos_feats
            elif self.append_pos:
                embeds = torch.cat((embeds[..., 1: -P], im, pos_feats), dim=-1)
            else:
                embeds = torch.cat((embeds[..., 1:], im), dim=-1)

        loss /= self.n_steps
        loss_full = self.l2_loss(pred.reshape(B, -1), yy.reshape(B, -1))

        return loss, loss_full, pred, pred_layer_list

    def training_step(self, batch, batch_idx):
        loss, loss_full, _, _ = self._learning_step(batch)
        self.log('train_loss', loss)
        self.log('train_loss_full', loss_full)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_full, preds, pred_list = self._learning_step(batch)
        self.log('valid_loss', loss)
        self.log('valid_loss_full', loss_full)

        if batch_idx == 0:
            xx, yy = batch
            expt = self.logger.experiment
            log_imshow(expt, xx[0, :, :, -1], 'gt t=9')
            log_imshow(expt, yy[0, :, :, -1], 'gt t=19')
            log_imshow(expt, preds[0, :, :, -1], 'pred t=19')

            layers = pred_list[-1]
            if layers:
                for i, layer in enumerate(layers):
                    log_imshow(expt, layer[0], f'layer {i} t=19')

    def test_step(self, batch, batch_idx):
        loss, loss_full, _, _ = self._learning_step(batch)
        self.log('test_loss', loss)
        self.log('test_loss_full', loss_full)


class MidpointNormalize(mpl.colors.Normalize):
    """See https://stackoverflow.com/a/50003503/3790116."""

    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(
            0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(
            1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [
            normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


def log_imshow(expt, tensor, name):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    vals = tensor.cpu().numpy()
    vmin = -3  # -1 if 'layer' in name else -3
    vmax = 3  # 1 if 'layer' in name else 3
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    im = ax.imshow(vals, interpolation='bilinear',
                   cmap=plt.get_cmap('bwr'), norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    fig.tight_layout()
    expt.log({f'{name}': wandb.Image(fig)})
    plt.close('all')
