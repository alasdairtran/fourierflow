import time
from typing import List, Optional

import elegy as eg
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from elegy.data import Dataset as ElegyDataset
from hydra.utils import instantiate
from jax_cfd.base.finite_differences import curl_2d
from jax_cfd.base.funcutils import repeated, trajectory
from jax_cfd.base.grids import Array, Grid
from jax_cfd.base.initial_conditions import filtered_velocity_field
from jax_cfd.spectral.time_stepping import crank_nicolson_rk4
from jax_cfd.spectral.utils import vorticity_to_velocity
from omegaconf import DictConfig
from torch.utils.data import Dataset as TorchDataset

from fourierflow.utils import downsample_vorticity_hat, import_string

from .base import Builder


class KolmogorovBuilder(Builder):
    name = 'kolmogorov'

    def __init__(self, train_path: str, valid_path: str, test_path: str,
                 train_k: int, valid_k: int, test_k: int, size: int,
                 loader_target: str = 'torch.utils.data.DataLoader', **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.train_dataset = KolmogorovMarkovDataset(train_path, size, train_k)
        self.valid_dataset = KolmogorovMarkovDataset(valid_path, size, valid_k)
        self.test_dataset = KolmogorovMarkovDataset(test_path, size, test_k)
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


class KolmogorovMarkovDataset(TorchDataset, ElegyDataset):
    def __init__(self, path, size, k):
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


class KolmogorovTrajectoryDataset(TorchDataset, ElegyDataset):
    def __init__(self, path, size, k):
        self.ds = xr.open_dataset(path)
        self.k = k
        self.B = len(self.ds.sample)

    def __len__(self):
        return self.B

    def __getitem__(self, b):
        ds = self.ds.isel(sample=b, time=slice(None, None, self.k))
        data = {
            'vx': ds.vx,
            'vy': ds.vy,
            'vorticity': ds.vorticity,
        }

        return {
            'data': data,
        }


def generate_kolmogorov(sim_grid: Grid,
                        out_sizes: List[int],
                        dt: float,
                        equation: DictConfig,
                        seed: jax.random.KeyArray,
                        vorticity0: Optional[Array] = None,
                        peak_wavenumber: float = 4.0,
                        max_velocity: float = 7.0,
                        inner_steps: int = 25,
                        outer_steps: int = 200,
                        warmup_steps: int = 40):
    """Generate 2D Kolmogorov flows, similar to Kochkov et al (2021).

    Adapted from https://github.com/google/jax-cfd/blob/main/notebooks/demo.ipynb
    """
    # Define the physical dimensions of the simulation.
    domain = ((0, 2 * jnp.pi), (0, 2 * jnp.pi))
    velocity_solve = vorticity_to_velocity(sim_grid)

    out_grids = {}
    for size in out_sizes:
        grid = Grid(shape=(size, size), domain=domain)
        out_grids[size] = grid

    if vorticity0 is None:
        # Construct a random initial velocity. The `filtered_velocity_field`
        # function ensures that the initial velocity is divergence free and it
        # filters out high frequency fluctuations.
        v0 = filtered_velocity_field(
            seed, sim_grid, max_velocity, peak_wavenumber)
        # Compute the fft of the vorticity. The spectral code assumes an fft'd
        # vorticity for an initial state.
        vorticity0 = curl_2d(v0).data

    vorticity_hat0 = jnp.fft.rfftn(vorticity0, axes=(0, 1))

    eqn = instantiate(equation)
    cnrk4 = crank_nicolson_rk4(eqn, dt)
    step_fn = repeated(cnrk4, inner_steps)

    # During warming up, we ignore intermediate results and just return
    # the final field
    if warmup_steps > 0:
        def ignore(_):
            return None
        trajectory_fn = trajectory(step_fn, warmup_steps, ignore)
        start = time.time()
        vorticity_hat0, _ = trajectory_fn(vorticity_hat0)
        elapsed = np.float32(time.time() - start)

        outs = {}
        for size, out_grid in out_grids.items():
            if size == sim_grid.shape[0]:
                vxhat, vyhat = velocity_solve(vorticity_hat0)
                out = {
                    'vx': jnp.fft.irfftn(vxhat, axes=(0, 1)),
                    'vy': jnp.fft.irfftn(vyhat, axes=(0, 1)),
                    'vorticity': jnp.fft.irfftn(vorticity_hat0, axes=(0, 1)),
                }
            else:
                out = downsample_vorticity_hat(
                    vorticity_hat0, velocity_solve, sim_grid, out_grid)
            outs[size] = out
        return outs, elapsed

    if outer_steps > 0:
        def downsample(vorticity_hat):
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

        trajectory_fn = trajectory(step_fn, outer_steps, downsample)
        start = time.time()
        _, trajs = trajectory_fn(vorticity_hat0)
        elapsed = np.float32(time.time() - start)

        return trajs, elapsed
