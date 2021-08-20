from typing import Any, Dict

import torch
from allennlp.common import Lazy
from allennlp.training.optimizers import Optimizer
from einops import rearrange, repeat

from fourierflow.modules import fourier_encode
from fourierflow.modules.loss import LpLoss
from fourierflow.registries import Experiment, Module, Scheduler
from fourierflow.viz import log_navier_stokes_heatmap


@Experiment.register('fourier_2d_scales')
class Fourier2DScalesExperiment(Experiment):
    def __init__(self,
                 #  conv_1: Module,
                 #  conv_2: Module,
                 conv_4: Module,
                 optimizer: Lazy[Optimizer],
                 n_steps: int,
                 max_freq: int = 32,
                 num_freq_bands: int = 8,
                 freq_base: int = 2,
                 scheduler: Lazy[Scheduler] = None,
                 scheduler_config: Dict[str, Any] = None,
                 low: float = 0,
                 high: float = 1,
                 use_fourier_position: bool = False):
        super().__init__()
        # self.conv_1 = conv_1
        # self.conv_2 = conv_2
        self.conv_4 = conv_4
        self.n_steps = n_steps
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

    def _calculate_loss(self, preds, data, n_steps):
        preds = rearrange(preds, '(b t) ... 1 -> b t ...', b=data.shape[0])
        # preds.shape == [batch_size, total_steps, *dim_sizes]

        # Ignore last `n_steps` steps as there are no targets
        preds = preds[:, :-n_steps]
        # preds.shape == [batch_size, time, *dim_sizes]

        preds = rearrange(preds, 'b t ... -> (b t) (...)')
        # preds.shape == [batch_size * time, product(*dim_sizes)]

        targets = data[:, ..., n_steps:]
        # targets.shape == [batch_size, *dim_sizes, time]

        # Batch up time dimension
        targets = rearrange(targets, 'b ... t -> (b t) (...)')
        # targets.shape == [batch_size * time, product(*dim_sizes)]

        loss = self.l2_loss(preds, targets)
        return loss

    def _train_for_step(self, data, xx, n, conv):
        # Ignore final n steps since there is no ground-truth output
        xx = xx[..., :-n, :]
        # xx.shape == [batch_size, *dim_sizes, total_steps - n, 3]

        # Batch up step dimension
        xx = rearrange(xx, 'b ... t e -> (b t) ... e')
        # xx.shape == [batch_size * time, *dim_sizes, 3]

        im, _ = conv(xx)
        # im.shape == [batch_size * time, *dim_sizes, 1]

        yy = data[:, ..., n:]
        # yy.shape == [batch_size, *dim_sizes, n_steps]

        # Batch up time dimension
        yy = rearrange(yy, 'b ... t -> (b t) ...')
        # yy.shape == [batch_size * time, *dim_sizes]

        BN = im.shape[0]
        loss = self.l2_loss(im.reshape(BN, -1), yy.reshape(BN, -1))

        return loss

    def _training_step(self, data):
        B, *dim_sizes, T = data.shape
        X, Y = dim_sizes
        # data.shape == [batch_size, *dim_sizes, total_steps]

        pos_feats = self.encode_positions(
            dim_sizes, self.low, self.high, self.use_fourier_position)
        # pos_feats.shape == [*dim_sizes, pos_size]

        pos_feats = repeat(pos_feats, '... -> b ...', b=B)
        # pos_feats.shape == [batch_size, *dim_sizes, n_dims]

        xx = repeat(data, '... -> ... 1')
        # xx.shape == [batch_size, *dim_sizes, total_steps, 1]

        all_pos_feats = repeat(pos_feats, '... e -> ... t e', t=T)

        xx = torch.cat([xx, all_pos_feats], dim=-1)
        # xx.shape == [batch_size, *dim_sizes, total_steps, 3]

        # loss_1 = self._train_for_step(data, xx, 1, self.conv_1)
        # self.log('train_loss_1', loss_1)
        # loss_2 = self._train_for_step(data, xx, 2, self.conv_2)
        # self.log('train_loss_2', loss_2)
        loss_19 = self._train_for_step(data, xx, 19, self.conv_4)
        self.log('train_loss_19', loss_19)
        # loss = (loss_1 + loss_2 + loss_4) / 3

        return loss_19

    def _valid_step(self, data, split):
        B, *dim_sizes, T = data.shape
        X, Y = dim_sizes
        # data.shape == [batch_size, *dim_sizes, total_steps]

        pos_feats = self.encode_positions(
            dim_sizes, self.low, self.high, self.use_fourier_position)
        # pos_feats.shape == [*dim_sizes, pos_size]

        pos_feats = repeat(pos_feats, '... -> b ...', b=B)
        # pos_feats.shape == [batch_size, *dim_sizes, n_dims]

        xx = repeat(data, '... -> ... 1')
        # xx.shape == [batch_size, *dim_sizes, total_steps, 1]

        all_pos_feats = repeat(pos_feats, '... e -> ... t e', t=T)

        xx = torch.cat([xx, all_pos_feats], dim=-1)
        # xx.shape == [batch_size, *dim_sizes, total_steps, 3]

        xx = xx[..., 0, :]
        # xx.shape == [batch_size, *dim_sizes, 3]

        yy = data[:, ..., -self.n_steps:]
        # yy.shape == [batch_size, *dim_sizes, n_steps]

        # Task, given only initial condition at time 0, predict time:
        # 1, 2, 4, 8, 15, 19
        # im_4, _ = self.conv_4(xx)
        # inputs_4 = torch.cat([im_4, pos_feats], dim=-1)

        # im_8, _ = self.conv_4(inputs_4)
        # inputs_8 = torch.cat([im_8, pos_feats], dim=-1)

        # im_12, _ = self.conv_4(inputs_8)
        # inputs_12 = torch.cat([im_12, pos_feats], dim=-1)

        # im_16, _ = self.conv_4(inputs_12)
        # inputs_16 = torch.cat([im_16, pos_feats], dim=-1)

        # im_18, _ = self.conv_2(inputs_16)
        # inputs_18 = torch.cat([im_18, pos_feats], dim=-1)

        im_19, _ = self.conv_4(xx)
        targets_19 = rearrange(data[:, ..., 19], 'b ... -> b (...)')
        loss_19 = self.l2_loss(im_19, targets_19)
        self.log(f'{split}_loss_19', loss_19)

        return im_19

    def training_step(self, batch, batch_idx):
        loss = self._training_step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        preds_19 = self._valid_step(batch, 'valid')

        if batch_idx == 0:
            data = batch
            expt = self.logger.experiment
            log_navier_stokes_heatmap(expt, data[0, :, :, 19], 'gt t=19')
            log_navier_stokes_heatmap(expt, preds_19[0, :, :, 0], 'pred t=19')

    def test_step(self, batch, batch_idx):
        self._valid_step(batch, 'test')
