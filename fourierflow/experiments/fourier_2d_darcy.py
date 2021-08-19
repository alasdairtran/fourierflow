from typing import Any, Dict

import torch
from allennlp.common import Lazy
from allennlp.training.optimizers import Optimizer
from einops import rearrange, repeat

from fourierflow.modules import fourier_encode
from fourierflow.modules.loss import LpLoss
from fourierflow.registry import Experiment, Module, Scheduler


@Experiment.register('fourier_2d_darcy')
class Fourier2DDarcyExperiment(Experiment):
    def __init__(self,
                 conv: Module,
                 optimizer: Lazy[Optimizer],
                 n_steps: int,
                 backcast_len: int = 10,
                 max_freq: int = 32,
                 num_freq_bands: int = 8,
                 freq_base: int = 2,
                 scheduler: Lazy[Scheduler] = None,
                 scheduler_config: Dict[str, Any] = None,
                 low: float = 0,
                 high: float = 1,
                 use_fourier_position: bool = False):
        super().__init__()
        self.conv = conv
        self.n_steps = n_steps
        self.backcast_len = backcast_len
        self.l2_loss = LpLoss(size_average=True)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config
        self.use_fourier_position = use_fourier_position
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.freq_base = freq_base
        self.low = low
        self.high = high
        self.register_buffer('_float', torch.FloatTensor([0.1]))

    def forward(self, x):
        x = self.conv(x)
        return x.squeeze()

    def encode_positions(self, dim_sizes, low=-1, high=1, fourier=True):
        # dim_sizes is a list of dimensions in all positional/time dimensions
        # e.g. for a 64 x 64 image over 20 steps, dim_sizes = [64, 64, 20]

        # A way to interpret `pos` is that we could append `pos` directly
        # to the raw inputs to attach the positional info to the raw features.
        def generate_grid(size):
            return torch.linspace(low, high, steps=size,
                                  device=self._float.device)
        grid_list = list(map(generate_grid, dim_sizes))
        pos = torch.stack(torch.meshgrid(*grid_list), dim=-1)
        # pos.shape == [*dim_sizes, n_dims]

        if not fourier:
            return pos

        # To get the fourier encodings, we will go one step further
        fourier_feats = fourier_encode(
            pos, self.max_freq, self.num_freq_bands, base=self.freq_base)
        # fourier_feats.shape == [*dim_sizes, n_dims, n_bands * 2 + 1]

        fourier_feats = rearrange(fourier_feats, '... n d -> ... (n d)')
        # fourier_feats.shape == [*dim_sizes, pos_size]

        return fourier_feats

    def _training_step(self, batch):
        x, y, mean, std = batch
        # x.shape == [batch_size, 85, 85]
        # y.shape == [batch_size, 85, 85]

        B, *dim_sizes = x.shape
        X, Y = dim_sizes
        # data.shape == [batch_size, *dim_sizes]

        pos_feats = self.encode_positions(
            dim_sizes, self.low, self.high, self.use_fourier_position)
        # pos_feats.shape == [*dim_sizes, pos_size]

        pos_feats = repeat(pos_feats, '... -> b ...', b=B)
        # pos_feats.shape == [batch_size, *dim_sizes, n_dims]

        # Add positional info
        xx = rearrange(x, 'b m n -> b m n 1')
        xx = torch.cat([xx, pos_feats], dim=-1)
        out = self.conv(xx)
        im = out[0]
        im = rearrange(im, 'b m n 1 -> b m n')
        im = (im * std) + mean
        # im.shape == [batch_size, *dim_sizes]

        loss = self.l2_loss(im.reshape(B, -1), y.reshape(B, -1))

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._training_step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._training_step(batch)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._training_step(batch)
        self.log('test_loss', loss)
