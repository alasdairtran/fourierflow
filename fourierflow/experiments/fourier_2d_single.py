from typing import Any, Dict

import torch
from allennlp.common import Lazy
from allennlp.training.optimizers import Optimizer
from einops import rearrange, repeat

from fourierflow.modules import Normalizer, fourier_encode
from fourierflow.modules.loss import LpLoss
from fourierflow.registries import Experiment, Module, Scheduler
from fourierflow.viz import log_navier_stokes_heatmap


@Experiment.register('fourier_2d_single')
class Fourier2DSingleExperiment(Experiment):
    def __init__(self,
                 conv: Module,
                 n_steps: int,
                 max_freq: int = 32,
                 num_freq_bands: int = 8,
                 freq_base: int = 2,
                 low: float = 0,
                 high: float = 1,
                 use_position: bool = True,
                 append_force: bool = False,
                 append_mu: bool = False,
                 max_accumulations: float = 1e6,
                 use_fourier_position: bool = False,
                 clip_val: float = 0.1,
                 noise_std: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.conv = conv
        self.n_steps = n_steps
        self.l2_loss = LpLoss(size_average=True)
        self.use_fourier_position = use_fourier_position
        self.use_position = use_position
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.freq_base = freq_base
        self.append_force = append_force
        self.append_mu = append_mu
        self.low = low
        self.high = high
        self.lr = None
        self.normalizer = Normalizer([conv.input_dim], max_accumulations)
        self.register_buffer('_float', torch.FloatTensor([0.1]))
        self.automatic_optimization = False  # activates manual optimization
        self.clip_val = clip_val
        self.noise_std = noise_std

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

    def _build_features(self, batch):
        x = batch['x']
        B, *dim_sizes, _ = x.shape
        X, Y = dim_sizes
        # data.shape == [batch_size, *dim_sizes]

        if self.use_position:
            pos_feats = self.encode_positions(
                dim_sizes, self.low, self.high, self.use_fourier_position)
            # pos_feats.shape == [*dim_sizes, pos_size]

            pos_feats = repeat(pos_feats, '... -> b ...', b=B)
            # pos_feats.shape == [batch_size, *dim_sizes, n_dims]

            x = torch.cat([x, pos_feats], dim=-1)
            # xx.shape == [batch_size, *dim_sizes, 3]

        if self.append_force:
            f = repeat(batch['f'], 'b m n -> b m n 1')
            x = torch.cat([x, f], dim=-1)

        if self.append_mu:
            mu = repeat(batch['mu'], 'b -> b m n 1', m=X, n=Y)
            x = torch.cat([x, mu], dim=-1)

        x = self.normalizer(x)

        x += torch.randn(*x.shape, device=x.device) * self.noise_std

        return x

    def _training_step(self, batch):
        x = self._build_features(batch)
        im = self.conv(x, global_step=self.global_step)['forecast']
        im = self.normalizer.inverse(im, channel=0)

        # im.shape == [batch_size * time, *dim_sizes, 1]

        BN = im.shape[0]
        loss = self.l2_loss(im.reshape(BN, -1), batch['y'].reshape(BN, -1))

        return loss

    def _valid_step(self, batch, split):
        data = batch['data']
        B, *dim_sizes, T = data.shape
        X, Y = dim_sizes
        # data.shape == [batch_size, *dim_sizes, total_steps]

        xx = repeat(data, '... -> ... 1')
        # xx.shape == [batch_size, *dim_sizes, total_steps, 1]

        if self.use_position:
            pos_feats = self.encode_positions(
                dim_sizes, self.low, self.high, self.use_fourier_position)
            # pos_feats.shape == [*dim_sizes, pos_size]

            pos_feats = repeat(pos_feats, '... -> b ...', b=B)
            # pos_feats.shape == [batch_size, *dim_sizes, n_dims]

            all_pos_feats = repeat(pos_feats, '... e -> ... t e', t=T)

            xx = torch.cat([xx, all_pos_feats], dim=-1)
            # xx.shape == [batch_size, *dim_sizes, total_steps, 3]

        xx = xx[..., -self.n_steps-1:-1, :]
        # xx.shape == [batch_size, *dim_sizes, n_steps, 3]

        if self.append_force:
            if len(batch['f'].shape) == 3:
                force = repeat(batch['f'], 'b m n -> b m n t 1',
                               t=xx.shape[-2])
            elif len(batch['f'].shape) == 4:
                f = batch['f'][..., -self.n_steps:]
                force = repeat(f, 'b m n t -> b m n t 1')

            xx = torch.cat([xx, force], dim=-1)

        if self.append_mu:
            mu = repeat(batch['mu'], 'b -> b m n t 1',
                        m=X, n=Y, t=xx.shape[-2])
            xx = torch.cat([xx, mu], dim=-1)

        yy = data[:, ..., -self.n_steps:]
        # yy.shape == [batch_size, *dim_sizes, n_steps]

        loss = 0
        # We predict one future one step at a time
        pred_layer_list = []
        for t in range(self.n_steps):
            if t == 0:
                x = xx[..., t, :]
            elif self.use_position and self.append_force and self.append_mu:
                x = torch.cat(
                    [im, pos_feats, force[..., t, :], mu[..., t, :]], dim=-1)
            elif self.use_position and self.append_force:
                x = torch.cat([im, pos_feats, force[..., t, :]], dim=-1)
            elif self.use_position:
                x = torch.cat([im, pos_feats], dim=-1)
            else:
                x = im
            # x.shape == [batch_size, *dim_sizes, 3]

            x = self.normalizer(x)
            out = self.conv(x)
            im = out['forecast']
            im = self.normalizer.inverse(im, channel=0)
            # im.shape == [batch_size, *dim_sizes, 1]

            y = yy[..., t]
            l = self.l2_loss(im.reshape(B, -1), y.reshape(B, -1))
            loss += l
            pred = im if t == 0 else torch.cat((pred, im), dim=-1)
            if 'forecast_list' in out:
                pred_layer_list.append(out['forecast_list'])

            if t == self.n_steps - 1:
                self.log(f'{split}_loss_last', l)

        loss /= self.n_steps
        loss_full = self.l2_loss(pred.reshape(B, -1), yy.reshape(B, -1))

        return loss, loss_full, pred, pred_layer_list

    def training_step(self, batch, batch_idx):
        # Accumulate normalization stats in the first epoch
        if self.current_epoch == 0:
            with torch.no_grad():
                self._build_features(batch)

        for i in range(self.conv.input_dim):
            self.log(f'normalizer_mean_{i}', self.normalizer.mean[i])
            self.log(f'normalizer_std_{i}', self.normalizer.std[i])

        if self.current_epoch >= 1:
            loss = self._training_step(batch)
            self.log('train_loss', loss, prog_bar=True)

            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            for group in opt.param_groups:
                torch.nn.utils.clip_grad_value_(group["params"], self.clip_val)
            opt.step()

            sch = self.lr_schedulers()
            sch.step()

            return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_full, preds, pred_list = self._valid_step(batch, 'valid')
        self.log('valid_loss_avg', loss)
        self.log('valid_loss', loss_full, prog_bar=True)

        if batch_idx == 0:
            data = batch['data']
            expt = self.logger.experiment
            log_navier_stokes_heatmap(expt, data[0, :, :, 9], 'gt t=9')
            log_navier_stokes_heatmap(expt, data[0, :, :, 19], 'gt t=19')
            log_navier_stokes_heatmap(expt, preds[0, :, :, -1], 'pred t=19')

            if pred_list:
                layers = pred_list[-1]
                for i, layer in enumerate(layers):
                    log_navier_stokes_heatmap(
                        expt, layer[0], f'layer {i} t=19')

    def test_step(self, batch, batch_idx):
        loss, loss_full, preds, pred_list = self._valid_step(
            batch, 'test')
        self.log('test_loss_avg', loss)
        self.log('test_loss', loss_full)
