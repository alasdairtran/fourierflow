import time
from functools import partial
from typing import Callable, List, Optional

import elegy as eg
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from elegy.data import Dataset as ElegyDataset
from hydra.utils import instantiate
from jax_cfd.base.boundaries import periodic_boundary_conditions
from jax_cfd.base.finite_differences import curl_2d
from jax_cfd.base.funcutils import repeated, trajectory
from jax_cfd.base.grids import Grid
from jax_cfd.base.initial_conditions import (filtered_velocity_field,
                                             wrap_velocities)
from jax_cfd.base.resize import downsample_staggered_velocity
from jax_cfd.spectral.utils import vorticity_to_velocity
from torch.utils.data import Dataset as TorchDataset

from fourierflow.utils import downsample_vorticity_hat, import_string

from .base import Builder

KEYS = ['vx', 'vy', 'vz']


class KolmogorovBuilder(Builder):
    name = 'kolmogorov'

    def __init__(self, train_dataset, valid_dataset, test_dataset,
                 loader_target: str = 'torch.utils.data.DataLoader', **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.DataLoader = import_string(loader_target)

    def train_dataloader(self) -> eg.data.DataLoader:
        loader = self.DataLoader(self.train_dataset,
                                 shuffle=True,
                                 **self.kwargs)
        return loader

    def val_dataloader(self) -> eg.data.DataLoader:
        loader = self.DataLoader(self.valid_dataset,
                                 shuffle=False,
                                 **self.kwargs)
        return loader

    def test_dataloader(self) -> eg.data.DataLoader:
        loader = self.DataLoader(self.test_dataset,
                                 shuffle=False,
                                 **self.kwargs)
        return loader


class KolmogorovElegyDataset(TorchDataset, ElegyDataset):
    def __init__(self, path, k):
        self.ds = xr.open_dataset(path)
        self.k = k
        self.B = len(self.ds.sample)
        self.T = len(self.ds.time) - self.k

    def __len__(self):
        return self.B * self.T

    def __getitem__(self, idx):
        b = idx // self.T
        t = idx % self.T
        k = self.k

        ds = self.ds.isel(sample=b, time=slice(t, t+k+1, k))
        in_ds = ds.isel(time=0)
        out_ds = ds.isel(time=1)

        inputs = {
            'vx': in_ds.vx,
            'vy': in_ds.vy,
            'vorticity': in_ds.vorticity,
        }

        outputs = {
            'vx': out_ds.vx,
            'vy': out_ds.vy,
            'vorticity': out_ds.vorticity,
        }

        return inputs, outputs


class KolmogorovTorchDataset(TorchDataset, ElegyDataset):
    def __init__(self, path, k):
        self.ds = xr.open_dataset(path)
        self.k = k
        self.B = len(self.ds.sample)
        self.T = len(self.ds.time) - self.k

    def __len__(self):
        return self.B * self.T

    def __getitem__(self, idx):
        b = idx // self.T
        t = idx % self.T
        k = self.k

        ds = self.ds.isel(sample=b, time=slice(t, t+k+1, k))
        in_ds = ds.isel(time=slice(0, 1)).transpose('x', 'y', 'time')
        out_ds = ds.isel(time=slice(1, 2)).transpose('x', 'y', 'time')

        return {
            'x': in_ds.vorticity.data,
            'y': out_ds.vorticity.data,
        }


class KolmogorovMultiTorchDataset(TorchDataset, ElegyDataset):
    def __init__(self, paths, k, batch_size):
        self.dss = [xr.open_dataset(path) for path in paths]
        self.k = k
        self.B = len(self.dss[0].sample)
        self.T = len(self.dss[0].time) - self.k
        self.counter = 0
        self.batch_size = batch_size
        self.ds_index = 0

    def __len__(self):
        return self.B * self.T

    def __getitem__(self, idx):
        b = idx // self.T
        t = idx % self.T
        k = self.k

        ds = self.dss[self.ds_index]
        ds = ds.isel(sample=b, time=slice(t, t+k+1, k))
        in_ds = ds.isel(time=slice(0, 1)).transpose('x', 'y', 'time')
        out_ds = ds.isel(time=slice(1, 2)).transpose('x', 'y', 'time')
        self.update_counter()

        return {
            'x': in_ds.vorticity.data,
            'y': out_ds.vorticity.data,
        }

    def update_counter(self):
        self.counter += 1
        if self.counter % self.batch_size == 0:
            self.ds_index = (self.ds_index + 1) % len(self.dss)


class KolmogorovTrajectoryDataset(TorchDataset, ElegyDataset):
    def __init__(self, init_path, path, corr_path, k, end=None):
        ds = xr.open_dataset(path)
        init_ds = xr.open_dataset(init_path)
        init_ds = init_ds.expand_dims(dim={'time': [0.0]})
        ds = xr.concat([init_ds, ds], dim='time')
        self.ds = ds.transpose('sample', 'x', 'y', 'time')

        corr_ds = xr.open_dataset(corr_path)
        self.corr_ds = corr_ds.transpose('sample', 'x', 'y', 'time')

        self.k = k
        self.B = len(self.ds.sample)
        self.end = end

    def __len__(self):
        return self.B

    def __getitem__(self, b):
        time_slice = slice(None, self.end, self.k)
        ds = self.ds.isel(sample=b, time=time_slice)
        corr_ds = self.corr_ds.isel(sample=b, time=time_slice)

        return {
            'times': ds.time.data,
            'data': ds.vorticity.data,
            'vx': ds.vx.data,
            'vy': ds.vy.data,
            'corr_data': corr_ds.vorticity.data,
        }


def generate_kolmogorov(sim_grid: Grid,
                        out_sizes: List[int],
                        method: str,
                        step_fn: Callable,
                        downsample_fn: Callable,
                        seed: jax.random.KeyArray,
                        initial_field: Optional[xr.Dataset] = None,
                        peak_wavenumber: float = 4.0,
                        max_velocity: float = 7.0,
                        inner_steps: int = 25,
                        outer_steps: int = 200,
                        warmup_steps: int = 40):
    """Generate 2D Kolmogorov flows, similar to Kochkov et al (2021).

    Adapted from https://github.com/google/jax-cfd/blob/main/notebooks/demo.ipynb
    """
    # Define the physical dimensions of the simulation.
    velocity_solve = vorticity_to_velocity(
        sim_grid) if sim_grid.ndim == 2 else None

    out_grids = {}
    for size in out_sizes:
        grid = Grid(shape=[size] * sim_grid.ndim, domain=sim_grid.domain)
        out_grids[size] = grid

    downsample = partial(downsample_fn, sim_grid, out_grids, velocity_solve)

    if initial_field is None:
        # Construct a random initial velocity. The `filtered_velocity_field`
        # function ensures that the initial velocity is divergence free and it
        # filters out high frequency fluctuations.
        v0 = filtered_velocity_field(
            seed, sim_grid, max_velocity, peak_wavenumber)
        if method == 'pseudo_spectral':
            # Compute the fft of the vorticity. The spectral code assumes an fft'd
            # vorticity for an initial state.
            vorticity0 = curl_2d(v0).data
    else:
        u, bcs = [], []
        for i in range(sim_grid.ndim):
            u.append(initial_field[KEYS[i]].data)
            bcs.append(periodic_boundary_conditions(sim_grid.ndim))
        v0 = wrap_velocities(u, sim_grid, bcs)
        if method == 'pseudo_spectral':
            vorticity0 = initial_field.vorticity.values

    if method == 'pseudo_spectral':
        state = jnp.fft.rfftn(vorticity0, axes=(0, 1))
    else:
        state = v0

    step_fn = instantiate(step_fn)
    outer_step_fn = repeated(step_fn, inner_steps)

    # During warming up, we ignore intermediate results and just return
    # the final field
    if warmup_steps > 0:
        def ignore(_):
            return None
        trajectory_fn = trajectory(outer_step_fn, warmup_steps, ignore)
        start = time.time()
        state, _ = trajectory_fn(state)
        elapsed = np.float32(time.time() - start)
        outs = downsample(state)
        return outs, elapsed

    if outer_steps > 0:
        start = time.time()
        trajectory_fn = trajectory(outer_step_fn, outer_steps, downsample)
        _, trajs = trajectory_fn(state)
        elapsed = np.float32(time.time() - start)
        return trajs, elapsed


def downsample_vorticity(sim_grid, out_grids, velocity_solve, vorticity_hat):
    outs = {}
    for size, out_grid in out_grids.items():
        if size == sim_grid.shape[0]:
            vxhat, vyhat = velocity_solve(vorticity_hat)
            out = {
                'vx': jnp.fft.irfftn(vxhat, axes=(0, 1)),
                'vy': jnp.fft.irfftn(vyhat, axes=(0, 1)),
                'vorticity': jnp.fft.irfftn(vorticity_hat, axes=(0, 1)),
            }
        else:
            out = downsample_vorticity_hat(
                vorticity_hat, velocity_solve, sim_grid, out_grid)
        outs[size] = out
    return outs


def downsample_velocity(sim_grid, out_grids, velocity_solve, u):
    outs = {}
    for size, out_grid in out_grids.items():
        out = {}
        if size == sim_grid.shape[0]:
            for i in range(sim_grid.ndim):
                out[KEYS[i]] = u[i].data
            if sim_grid.ndim == 2:
                out['vorticity'] = curl_2d(u).data
        else:
            u_new = downsample_staggered_velocity(
                sim_grid, out_grid, u)
            for i in range(sim_grid.ndim):
                out[KEYS[i]] = u_new[i].data
            if sim_grid.ndim == 2:
                out['vorticity'] = curl_2d(u_new).data
        outs[size] = out
    return outs
