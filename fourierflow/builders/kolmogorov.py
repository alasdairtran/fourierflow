import time
from typing import Dict, List, Optional, cast

import elegy as eg
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from elegy.data import Dataset as ElegyDataset
from jax_cfd.base.finite_differences import curl_2d
from jax_cfd.base.funcutils import repeated, trajectory
from jax_cfd.base.grids import Array, Grid
from jax_cfd.base.initial_conditions import filtered_velocity_field
from jax_cfd.spectral.time_stepping import crank_nicolson_rk4
from jax_cfd.spectral.utils import vorticity_to_velocity
from omegaconf import DictConfig, OmegaConf
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
        ds = xr.open_dataset(path)
        self.vorticity = ds.vorticity
        self.k = k

        self.B = self.vorticity.shape[0]
        self.T = self.vorticity.shape[1] - self.k
        init_size = self.vorticity.shape[2]

        domain = ((0, 2 * jnp.pi), (0, 2 * jnp.pi))
        self.init_grid = Grid(shape=(init_size, init_size), domain=domain)
        self.out_grid = Grid(shape=(size, size), domain=domain)
        self.velocity_solve = vorticity_to_velocity(self.init_grid)

    def __len__(self):
        return self.B * self.T

    def __getitem__(self, idx):
        b = idx // self.T
        t = idx % self.T
        k = self.k

        vorticity = self.vorticity[b, t:t+k+1:k].values
        vorticity_hat = jnp.fft.rfftn(vorticity, axes=(1, 2))
        x = downsample_vorticity_hat(vorticity_hat[0], self.velocity_solve,
                                     self.init_grid, self.out_grid)['vorticity']
        y = downsample_vorticity_hat(vorticity_hat[1], self.velocity_solve,
                                     self.init_grid, self.out_grid)['vorticity']

        return {
            'x': x,
            'y': y,
        }


class KolmogorovTrajectoryDataset(TorchDataset, ElegyDataset):
    def __init__(self, path, size, k):
        ds = xr.open_dataset(path)
        self.vorticity = ds.vorticity

        self.k = k
        self.B = self.data.shape[0]
        init_size = self.vorticity.shape[2]

        domain = ((0, 2 * jnp.pi), (0, 2 * jnp.pi))
        self.init_grid = Grid(shape=(init_size, init_size), domain=domain)
        self.out_grid = Grid(shape=(size, size), domain=domain)
        self.velocity_solve = vorticity_to_velocity(self.init_grid)

    def __len__(self):
        return self.B

    def __getitem__(self, b):
        vorticity = self.vorticity[b, ::self.k].values
        vorticity_hat = jnp.fft.rfftn(vorticity, axes=(1, 2))
        data = downsample_vorticity_hat(vorticity_hat[0], self.velocity_solve,
                                        self.init_grid, self.out_grid)['vorticity']
        return {
            'data': data,
        }


def generate_kolmogorov(sim_size: int,
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
    sim_grid = Grid(shape=(sim_size, sim_size), domain=domain)
    velocity_solve = vorticity_to_velocity(sim_grid)

    out_grids = {}
    for out_size in out_sizes:
        grid = Grid(shape=(out_size, out_size), domain=domain)
        out_grids[out_size] = grid

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

    Equation = import_string(equation.target)
    kwargs = cast(Dict, OmegaConf.to_object(equation.kwargs))
    if 'forcing_fn' in kwargs:
        kwargs['forcing_fn'] = import_string(kwargs['forcing_fn'])
    eqn = Equation(grid=sim_grid, **kwargs)
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
            if size == sim_size:
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
                if size == sim_size:
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
        _, traj = trajectory_fn(vorticity_hat0)
        elapsed = np.float32(time.time() - start)

        return traj, elapsed
