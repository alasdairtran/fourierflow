import math
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch
import wandb
import xarray as xr
from einops import rearrange, repeat
from jax_cfd.base.grids import Grid
from jax_cfd.spectral.utils import vorticity_to_velocity
from torch import nn

from fourierflow.modules import Normalizer, fourier_encode
from fourierflow.modules.loss import LpLoss
from fourierflow.utils import downsample_vorticity, downsample_vorticity_hat

from .base import Routine


class Grid2DMarkovExperiment(Routine):
    def __init__(self,
                 conv: nn.Module,
                 n_steps: Optional[int] = None,
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
                 accumulate_grad_batches: int = 1,
                 noise_std: float = 0.0,
                 shuffle_grid: bool = False,
                 use_velocity: bool = False,
                 learn_difference: bool = False,
                 step_size: float = 1.0,
                 n_test_steps_logged: Optional[int] = None,
                 domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)),
                 heatmap_scale: int = 1,
                 pred_path: Optional[Path] = None,
                 grid_size: int = 64,
                 **kwargs):
        super().__init__(**kwargs)
        self.conv = conv
        self.n_steps = n_steps
        self.l2_loss = LpLoss(size_average=True)
        self.use_fourier_position = use_fourier_position
        self.use_position = use_position
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
        self.accumulate_grad_batches = accumulate_grad_batches
        self.clip_val = clip_val
        self.noise_std = noise_std
        self.shuffle_grid = shuffle_grid
        self.use_velocity = use_velocity
        self.learn_difference = learn_difference
        self.step_size = step_size
        self.n_test_steps_logged = n_test_steps_logged
        self.heatmap_scale = heatmap_scale
        self.domain = domain
        self.pred_path = pred_path
        if self.shuffle_grid:
            self.x_idx = torch.randperm(grid_size)
            self.x_inv = torch.argsort(self.x_idx)
            self.y_idx = torch.randperm(grid_size)
            self.y_inv = torch.argsort(self.y_idx)

        if self.use_velocity:
            jax.config.update('jax_platform_name', 'cpu')
            grid = Grid(shape=(grid_size, grid_size), domain=self.domain)
            kx, ky = grid.rfft_mesh()
            two_pi_i = 2 * jnp.pi * 1j
            lap = two_pi_i ** 2 * (abs(kx)**2 + abs(ky)**2)
            lap = lap.at[0, 0].set(1)

            self.register_buffer('kx', torch.from_numpy(np.array(kx)))
            self.register_buffer('ky', torch.from_numpy(np.array(ky)))
            self.register_buffer('lap', torch.from_numpy(np.array(lap)))

    def forward(self, data):
        return self._valid_step(data)

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
            omega_hat = torch.fft.rfftn(x, dim=[1, 2], norm='backward')
            psi_hat = -omega_hat / repeat(self.lap, 'm n -> b m n 1', b=B)
            ky = repeat(self.ky, 'm n -> b m n 1', b=B)
            kx = repeat(self.kx, 'm n -> b m n 1', b=B)

            # Velocity field in x-direction = psi_y
            q = 2 * math.pi * 1j * ky * psi_hat
            q = torch.fft.irfftn(q, dim=[1, 2], norm='backward')

            # Velocity field in y-direction = -psi_x
            v = -2 * math.pi * 1j * kx * psi_hat
            v = torch.fft.irfftn(v, dim=[1, 2], norm='backward')

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
            w_hat = torch.fft.rfftn(inputs, dim=[1, 2], norm='backward')
            psi_hat = -w_hat / repeat(self.lap, 'm n -> b m n t 1', b=B, t=T)
            ky = repeat(self.ky, 'm n -> b m n t 1', b=B, t=T)
            kx = repeat(self.kx, 'm n -> b m n t 1', b=B, t=T)

            # Velocity field in x-direction = psi_y
            q = 2 * math.pi * 1j * ky * psi_hat
            q = torch.fft.irfftn(q, dim=[1, 2], norm='backward')

            # Velocity field in y-direction = -psi_x
            v = -2 * math.pi * 1j * kx * psi_hat
            v = torch.fft.irfftn(v, dim=[1, 2], norm='backward')

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
                    w_hat = torch.fft.rfftn(im, dim=[1, 2], norm='backward')
                    psi_hat = -w_hat / \
                        repeat(self.lap, 'm n -> b m n t', b=B, t=im.shape[-1])
                    ky = repeat(self.ky, 'm n -> b m n t',
                                b=B, t=im.shape[-1])
                    kx = repeat(self.kx, 'm n -> b m n t',
                                b=B, t=im.shape[-1])

                    # Velocity field in x-direction = psi_y
                    q = 2 * math.pi * 1j * ky * psi_hat
                    q = torch.fft.irfftn(q, dim=[1, 2], norm='backward')

                    # Velocity field in y-direction = -psi_x
                    v = -2 * math.pi * 1j * kx * psi_hat
                    v = torch.fft.irfftn(v, dim=[1, 2], norm='backward')

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

        return loss, step_losses, preds, pred_layer_list

    def compute_losses(self, batch, loss, preds):
        data = batch['data']
        B, *dim_sizes, T = data.shape
        n_steps = self.n_steps or T - 1
        yy = data[:, ..., -n_steps:]
        times = batch['times'][0, -n_steps:]
        loss /= n_steps
        loss_full = self.l2_loss(preds.reshape(
            B, -1), yy.reshape(B, -1))

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

        # We reduce all grid sizes to 32x32 before computing correlation
        reduced_time_until = time_until
        p_2 = p
        if 'corr_data' in batch:
            corr_yy = batch['corr_data'][:, ..., -n_steps:]

            corr_size = corr_yy.shape[1]
            if dim_sizes[0] != corr_size:
                jax.config.update('jax_platform_name', 'cpu')
                preds_2 = downsample_vorticity(preds, corr_size, self.domain)
                preds_2 = preds_2.to(preds.device)
                pred_2_norm = torch.norm(preds_2, dim=[1, 2], keepdim=True)
                corr_yy_norm = torch.norm(corr_yy, dim=[1, 2], keepdim=True)
                p_2 = (preds_2 / pred_2_norm) * (corr_yy / corr_yy_norm)
                p_2 = p_2.sum(dim=[1, 2]).mean(dim=0)

                has_diverged = p_2 < 0.95
                diverged_idx = has_diverged.nonzero()
                diverged_t = diverged_idx[0, 0] if len(
                    diverged_idx) > 0 else len(has_diverged)
                reduced_time_until = diverged_t * self.step_size

        return loss, loss_full, time_until, reduced_time_until, p_2, times

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
                if self.accumulate_grad_batches == 1:
                    opt = self.optimizers()
                    opt.zero_grad()
                    self.manual_backward(loss)
                    if self.clip_val:
                        for group in opt.param_groups:
                            torch.nn.utils.clip_grad_value_(group["params"],
                                                            self.clip_val)
                    opt.step()

                else:
                    opt = self.optimizers()
                    loss /= self.accumulate_grad_batches
                    self.manual_backward(loss)
                    if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                        if self.clip_val:
                            for group in opt.param_groups:
                                torch.nn.utils.clip_grad_value_(group["params"],
                                                                self.clip_val)
                        opt.step()
                        opt.zero_grad()

                sch = self.lr_schedulers()
                sch.step()

            return loss

    def validation_step(self, batch, batch_idx):
        loss, step_losses, preds, pred_list = self._valid_step(batch)
        loss, loss_full, time_until, reduced_time_until, p, times = self.compute_losses(
            batch, loss, preds)

        if torch.isnan(loss):
            loss = 9999

        self.log('valid_loss_avg', loss)
        self.log('valid_loss', loss_full, prog_bar=True)
        self.log('valid_time_until', time_until, prog_bar=True)
        self.log('valid_reduced_time_until', reduced_time_until)
        self.log('valid_corr', p.mean())

    def test_step(self, batch, batch_idx):
        loss, step_losses, preds, pred_layer_list = self._valid_step(batch)
        loss, loss_full, time_until, reduced_time_until, p, times = self.compute_losses(
            batch, loss, preds)
        self.log('test_loss_avg', loss)
        self.log('test_loss', loss_full)
        self.log('test_time_until', time_until)
        self.log('test_reduced_time_until', reduced_time_until)
        self.log('test_corr', p.mean())

        if self.logger:
            corr_rows = list(zip(times.cpu().numpy(), p.cpu().numpy()))
            self.logger.experiment.log({
                'test_correlations': wandb.Table(['time', 'corr'], corr_rows)})

            loss_rows = list(zip(times.cpu().numpy(), step_losses))
            self.logger.experiment.log({
                'test_losses': wandb.Table(['time', 'loss'], loss_rows)})

        if self.pred_path:
            vorticities = preds.cpu().numpy()
            B, X, Y, T = vorticities.shape
            vorticities_hat = jnp.fft.rfftn(vorticities, axes=(1, 2))

            sim_grid = Grid(shape=[X, Y], domain=self.domain)
            out_grid = Grid(shape=[64, 64], domain=self.domain)
            velocity_solve = vorticity_to_velocity(sim_grid)

            vxs, vys, ws = [], [], []
            for b in range(B):
                vxb, vyb, wb = [], [], []
                for t in range(T):
                    vorticity_hat = vorticities_hat[b, ..., t]

                    if X > 64:
                        out = downsample_vorticity_hat(
                            vorticity_hat, velocity_solve, sim_grid, out_grid)
                        vxb.append(out['vx'])
                        vyb.append(out['vy'])
                        wb.append(out['vorticity'])
                    else:
                        vxhat, vyhat = velocity_solve(vorticity_hat)
                        vxb.append(jnp.fft.irfftn(vxhat, axes=(0, 1)))
                        vyb.append(jnp.fft.irfftn(vyhat, axes=(0, 1)))
                        wb.append(vorticities[b, ..., t])

                vxs.append(jnp.stack(vxb, axis=-1))
                vys.append(jnp.stack(vyb, axis=-1))
                ws.append(jnp.stack(wb, axis=-1))

            vxs = jnp.stack(vxs, axis=0)
            vys = jnp.stack(vys, axis=0)
            ws = jnp.stack(ws, axis=0)

            dim_names = ('sample', 'x', 'y', 'time')
            data_vars = {
                'vorticity': (dim_names, ws),
                'vx': (dim_names, vxs),
                'vy': (dim_names, vys),
            }
            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    'sample': range(B),
                    'time': times.cpu().numpy(),
                    'x': out_grid.axes()[0],
                    'y': out_grid.axes()[1],
                })
            ds.to_netcdf(self.pred_path, engine='h5netcdf')

        # if self.n_test_steps_logged is None:
        #     length = len(step_losses)
        # else:
        #     length = self.n_test_steps_logged
        # for i in range(length):
        #     self.log(f'test_loss_{i}', step_losses[i])
