import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat

from fourierflow.modules import Normalizer, fourier_encode
from fourierflow.modules.loss import LpLoss
from fourierflow.viz import log_navier_stokes_heatmap

from .base import Routine


class Grid2DMarkovExperiment(Routine):
    def __init__(self,
                 conv: nn.Module,
                 n_steps: Optional[int] = None,
                 k_max: int = 32,
                 num_freq_bands: int = 8,
                 freq_base: int = 2,
                 low: float = 0,
                 high: float = 1,
                 use_position: bool = True,
                 append_force: bool = False,
                 append_mu: bool = False,
                 max_accumulations: float = 1e6,
                 should_normalize: bool = True,
                 use_fourier_position: bool = False,
                 clip_val: Optional[float] = 0.1,
                 automatic_optimization: bool = False,
                 noise_std: float = 0.0,
                 shuffle_grid: bool = False,
                 use_velocity: bool = False,
                 learn_difference: bool = False,
                 step_size: float = 1/64,
                 n_test_steps_logged: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.conv = conv
        self.n_steps = n_steps
        self.l2_loss = LpLoss(size_average=True)
        self.use_fourier_position = use_fourier_position
        self.use_position = use_position
        self.k_max = k_max
        self.num_freq_bands = num_freq_bands
        self.freq_base = freq_base
        self.append_force = append_force
        self.append_mu = append_mu
        self.low = low
        self.high = high
        self.lr = None
        self.should_normalize = should_normalize
        self.normalizer = Normalizer([conv.input_dim], max_accumulations)
        self.register_buffer('_float', torch.FloatTensor([0.1]))
        self.automatic_optimization = automatic_optimization
        self.clip_val = clip_val
        self.noise_std = noise_std
        self.shuffle_grid = shuffle_grid
        self.use_velocity = use_velocity
        self.learn_difference = learn_difference
        self.step_size = step_size
        self.n_test_steps_logged = n_test_steps_logged
        if self.shuffle_grid:
            self.x_idx = torch.randperm(64)
            self.x_inv = torch.argsort(self.x_idx)
            self.y_idx = torch.randperm(64)
            self.y_inv = torch.argsort(self.y_idx)

        if self.use_velocity:
            # Wavenumbers in y-direction
            k_y = torch.cat((
                torch.arange(start=0, end=k_max, step=1),
                torch.arange(start=-k_max, end=0, step=1)),
                0).repeat(64, 1)
            # Wavenumbers in x-direction
            k_x = k_y.transpose(0, 1)
            # Negative Laplacian in Fourier space
            lap = 4 * (math.pi**2) * (k_x**2 + k_y**2)
            lap[0, 0] = 1.0
            self.register_buffer('k_x', k_x)
            self.register_buffer('k_y', k_y)
            self.register_buffer('lap', lap)

    def forward(self, data):
        batch = {'data': data}
        return self._valid_step(batch)

    def encode_positions(self, dim_sizes, low=-1, high=1, fourier=True):
        # dim_sizes is a list of dimensions in all positional/time dimensions
        # e.g. for a 64 x 64 image over 20 steps, dim_sizes = [64, 64, 20]

        # A way to interpret `pos` is that we could append `pos` directly
        # to the raw inputs to attach the positional info to the raw features.
        def generate_grid(size):
            return torch.linspace(low, high, steps=size,
                                  device=self._float.device)
        grid_list = list(map(generate_grid, dim_sizes))
        pos = torch.stack(torch.meshgrid(*grid_list, indexing='ij'), dim=-1)
        # pos.shape == [*dim_sizes, n_dims]

        if not fourier:
            return pos

        # To get the fourier encodings, we will go one step further
        fourier_feats = fourier_encode(
            pos, self.k_max, self.num_freq_bands, base=self.freq_base)
        # fourier_feats.shape == [*dim_sizes, n_dims, n_bands * 2 + 1]

        fourier_feats = rearrange(fourier_feats, '... n d -> ... (n d)')
        # fourier_feats.shape == [*dim_sizes, pos_size]

        return fourier_feats

    def _build_features(self, batch):
        x = batch['x']
        B, *dim_sizes, _ = x.shape
        X, Y = dim_sizes
        # data.shape == [batch_size, *dim_sizes]

        if self.use_velocity:
            omega_hat = torch.fft.fftn(x, dim=[1, 2], norm='backward')
            psi_hat = omega_hat / repeat(self.lap, 'm n -> b m n 1', b=B)
            k_y = repeat(self.k_y, 'm n -> b m n 1', b=B)
            k_x = repeat(self.k_x, 'm n -> b m n 1', b=B)

            # Velocity field in x-direction = psi_y
            q = psi_hat.clone()
            q_real_temp = q.real.clone()
            q.real = -2 * math.pi * k_y * q.imag
            q.imag = 2 * math.pi * k_y * q_real_temp
            q = torch.fft.ifftn(q, dim=[1, 2], norm='backward').real

            # Velocity field in y-direction = -psi_x
            v = psi_hat.clone()
            v_real_temp = v.real.clone()
            v.real = 2 * math.pi * k_x * v.imag
            v.imag = -2 * math.pi * k_x * v_real_temp
            v = torch.fft.ifftn(v, dim=[1, 2], norm='backward').real

            x = torch.cat([x, q, v], dim=-1)

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

        if self.should_normalize:
            x = self.normalizer(x)

        x += torch.randn(*x.shape, device=x.device) * self.noise_std

        return x

    def _training_step(self, batch):
        x = self._build_features(batch)
        B, M, N, _ = x.shape
        # x.shape == [batch_size, *dim_sizes, input_size]

        if self.shuffle_grid:
            x = x[:, self.x_idx][:, :, self.y_idx]

        im = self.conv(x, global_step=self.global_step)['forecast']

        if self.shuffle_grid:
            im = im[:, :, self.y_inv][:, self.x_inv]
        if self.should_normalize:
            im = self.normalizer.inverse(im, channel=0)

        # im.shape == [batch_size * time, *dim_sizes, 1]

        BN = im.shape[0]
        targets = batch['dy'] if self.learn_difference else batch['y']
        loss = self.l2_loss(im.reshape(BN, -1), targets.reshape(BN, -1))

        return loss

    def _valid_step(self, batch):
        data = batch['data']
        inputs = data

        B, *dim_sizes, T = inputs.shape
        X, Y = dim_sizes
        # data.shape == [batch_size, *dim_sizes, total_steps]

        inputs = repeat(inputs, '... -> ... 1')
        # inputs.shape == [batch_size, *dim_sizes, total_steps, 1]

        if self.use_velocity:
            w_hat = torch.fft.fftn(inputs, dim=[1, 2], norm='backward')
            psi_hat = w_hat / repeat(self.lap, 'm n -> b m n t 1', b=B, t=T)
            k_y = repeat(self.k_y, 'm n -> b m n t 1', b=B, t=T)
            k_x = repeat(self.k_x, 'm n -> b m n t 1', b=B, t=T)

            # Velocity field in x-direction = psi_y
            q = psi_hat.clone()
            q_real_temp = q.real.clone()
            q.real = -2 * math.pi * k_y * q.imag
            q.imag = 2 * math.pi * k_y * q_real_temp
            q = torch.fft.ifftn(q, dim=[1, 2], norm='backward').real

            # Velocity field in y-direction = -psi_x
            v = psi_hat.clone()
            v_real_temp = v.real.clone()
            v.real = 2 * math.pi * k_x * v.imag
            v.imag = -2 * math.pi * k_x * v_real_temp
            v = torch.fft.ifftn(v, dim=[1, 2], norm='backward').real

            inputs = torch.cat([inputs, q, v], dim=-1)

        if self.use_position:
            pos_feats = self.encode_positions(
                dim_sizes, self.low, self.high, self.use_fourier_position)
            # pos_feats.shape == [*dim_sizes, pos_size]

            pos_feats = repeat(pos_feats, '... -> b ...', b=B)
            # pos_feats.shape == [batch_size, *dim_sizes, n_dims]

            all_pos_feats = repeat(pos_feats, '... e -> ... t e', t=T)

            inputs = torch.cat([inputs, all_pos_feats], dim=-1)
            # inputs.shape == [batch_size, *dim_sizes, total_steps, 3]

        n_steps = self.n_steps or T - 1
        inputs = inputs[..., -n_steps-1:-1, :]
        # inputs.shape == [batch_size, *dim_sizes, n_steps, 3]

        xx = inputs

        if self.append_force:
            if len(batch['f'].shape) == 3:
                force = repeat(batch['f'], 'b m n -> b m n t 1',
                               t=xx.shape[-2])
            elif len(batch['f'].shape) == 4:
                f = batch['f'][..., -n_steps:]
                force = repeat(f, 'b m n t -> b m n t 1')

            xx = torch.cat([xx, force], dim=-1)

        if self.append_mu:
            mu = repeat(batch['mu'], 'b -> b m n t 1',
                        m=X, n=Y, t=xx.shape[-2])
            xx = torch.cat([xx, mu], dim=-1)

        yy = data[:, ..., -n_steps:]
        # yy.shape == [batch_size, *dim_sizes, n_steps]

        loss = 0
        step_losses = []
        # We predict one future one step at a time
        pred_layer_list = []
        for t in range(n_steps):
            if t == 0:
                x = xx[..., t, :]
                prev_im = x[..., 0:1]
            else:
                if self.use_velocity:
                    w_hat = torch.fft.fftn(im, dim=[1, 2], norm='backward')
                    psi_hat = w_hat / \
                        repeat(self.lap, 'm n -> b m n t', b=B, t=im.shape[-1])
                    k_y = repeat(self.k_y, 'm n -> b m n t',
                                 b=B, t=im.shape[-1])
                    k_x = repeat(self.k_x, 'm n -> b m n t',
                                 b=B, t=im.shape[-1])

                    # Velocity field in x-direction = psi_y
                    q = psi_hat.clone()
                    q_real_temp = q.real.clone()
                    q.real = -2 * math.pi * k_y * q.imag
                    q.imag = 2 * math.pi * k_y * q_real_temp
                    q = torch.fft.ifftn(q, dim=[1, 2], norm='backward').real

                    # Velocity field in y-direction = -psi_x
                    v = psi_hat.clone()
                    v_real_temp = v.real.clone()
                    v.real = 2 * math.pi * k_x * v.imag
                    v.imag = -2 * math.pi * k_x * v_real_temp
                    v = torch.fft.ifftn(v, dim=[1, 2], norm='backward').real

                    im = torch.cat([im, q, v], dim=-1)
                if self.use_position:
                    im = torch.cat([im, pos_feats], dim=-1)
                if self.append_force:
                    im = torch.cat([im, force[..., t, :]], dim=-1)
                if self.append_mu:
                    im = torch.cat([im, mu[..., t, :]], dim=-1)
                x = im
            # x.shape == [batch_size, *dim_sizes, 3]

            if self.should_normalize:
                x = self.normalizer(x)
            if self.shuffle_grid:
                x = x[:, self.x_idx][:, :, self.y_idx]

            out = self.conv(x)
            im = out['forecast']

            if self.shuffle_grid:
                im = im[:, :, self.y_inv][:, self.x_inv]
            if self.should_normalize:
                im = self.normalizer.inverse(im, channel=0)
            # im.shape == [batch_size, *dim_sizes, 1]

            if self.learn_difference:
                y = yy[..., t] - yy[..., t-1]
            else:
                y = yy[..., t]
            l = self.l2_loss(im.reshape(B, -1), y.reshape(B, -1))
            step_losses.append(l)
            loss += l
            if self.learn_difference:
                im = prev_im + im
                prev_im = im
            preds = im if t == 0 else torch.cat((preds, im), dim=-1)
            if 'forecast_list' in out:
                pred_layer_list.append(out['forecast_list'])

        # preds.shape == [batch_size, *dim_sizes, n_steps]
        # yy.shape == [batch_size, *dim_sizes, n_steps]

        pred_norm = torch.norm(preds, dim=[1, 2], keepdim=True)
        yy_norm = torch.norm(yy, dim=[1, 2], keepdim=True)
        p = (preds / pred_norm) * (yy / yy_norm)
        p = p.sum(dim=[1, 2]).mean(dim=0)
        # p.shape == [n_steps]

        has_diverged = p < 0.95
        diverged_idx = has_diverged.nonzero()
        diverged_t = diverged_idx[0, 0] if len(
            diverged_idx) > 0 else len(has_diverged)
        time_until = diverged_t * self.step_size

        loss /= n_steps
        loss_full = self.l2_loss(preds.reshape(
            B, -1), yy.reshape(B, -1))

        return loss, loss_full, preds, pred_layer_list, step_losses, time_until

    def training_step(self, batch, batch_idx):
        # Accumulate normalization stats in the first epoch
        if self.should_normalize and self.current_epoch == 0:
            with torch.no_grad():
                self._build_features(batch)

        if self.should_normalize:
            for i in range(self.conv.input_dim):
                self.log(f'normalizer_mean_{i}', self.normalizer.mean[i])
                self.log(f'normalizer_std_{i}', self.normalizer.std[i])

        if not self.should_normalize or self.current_epoch >= 1:
            loss = self._training_step(batch)
            self.log('train_loss', loss, prog_bar=True)

            if not self.automatic_optimization:
                opt = self.optimizers()
                opt.zero_grad()
                self.manual_backward(loss)
                if self.clip_val:
                    for group in opt.param_groups:
                        torch.nn.utils.clip_grad_value_(group["params"],
                                                        self.clip_val)
                opt.step()

                sch = self.lr_schedulers()
                sch.step()

            return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_full, preds, pred_list, _, time_until = self._valid_step(
            batch)
        self.log('valid_loss_avg', loss)
        self.log('valid_loss', loss_full, prog_bar=True)
        self.log('valid_time_until', time_until, prog_bar=True)

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
        loss, loss_full, _, _, step_losses, time_until = self._valid_step(
            batch)
        self.log('test_loss_avg', loss)
        self.log('test_loss', loss_full)
        self.log('test_time_until', time_until, prog_bar=True)
        for i in range(self.n_test_steps_logged or len(step_losses)):
            self.log(f'test_loss_{i}', step_losses[i])
