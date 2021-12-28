import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import jax_cfd.data.xarray_utils as xru
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from .base import Builder


class KolmogorovBuilder(Builder):
    name = 'kolmogorov'

    def __init__(self, train_path: str, valid_path: str, test_path: str, train_k: int,
                 valid_k: int, test_k: int, n_workers: int, batch_size: int):
        super().__init__()
        self.n_workers = n_workers
        self.batch_size = batch_size

        train_ds = xr.open_dataset(train_path)
        train_ds['vorticity'] = xru.vorticity_2d(train_ds)
        train_w = train_ds['vorticity'].values
        train_w = train_w.transpose(0, 2, 3, 1)
        # train_w.shape == [32, 64, 64, 4880]

        valid_ds = xr.open_dataset(valid_path)
        valid_ds['vorticity'] = xru.vorticity_2d(valid_ds)
        valid_w = valid_ds['vorticity'].values
        valid_w = valid_w.transpose(0, 2, 3, 1)
        # valid_w.shape == [32, 64, 64, 488]

        test_ds = xr.open_dataset(test_path)
        test_ds['vorticity'] = xru.vorticity_2d(test_ds)
        test_w = test_ds['vorticity'].values
        test_w = test_w.transpose(0, 2, 3, 1)
        # valid_w.shape == [32, 64, 64, 488]

        self.train_dataset = NavierStokesTrainingDataset(train_w, train_k)
        self.valid_dataset = NavierStokesDataset(valid_w, valid_k)
        self.test_dataset = NavierStokesDataset(test_w, test_k)

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.n_workers,
                            drop_last=False,
                            pin_memory=True)
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(self.valid_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.n_workers,
                            drop_last=False,
                            pin_memory=True)
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(self.test_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.n_workers,
                            drop_last=False,
                            pin_memory=True)
        return loader


class NavierStokesTrainingDataset(Dataset):
    def __init__(self, data, k):
        self.data = data
        self.k = k

        self.B = self.data.shape[0]
        self.T = self.data.shape[-1] - self.k

    def __len__(self):
        return self.B * self.T

    def __getitem__(self, idx):
        b = idx // self.T
        t = idx % self.T
        return {
            'x': self.data[b, :, :, t:t+1],
            'y': self.data[b, :, :, t+self.k:t+self.k+1],
        }


class NavierStokesDataset(Dataset):
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.B = self.data.shape[0]

    def __len__(self):
        return self.B

    def __getitem__(self, b):
        return {
            'data': self.data[b, :, :, ::self.k],
        }


def generate_kolmogorov(size: int,
                        dt: float,
                        density: float = 1.0,
                        viscosity: float = 1e-3,
                        max_velocity: float = 2.0,
                        seed: int = 0,
                        inner_steps: int = 25,
                        outer_steps: int = 200):
    """Generate 2D Kolmogorov flows, similar to Kochkov et al (2021).

    Adapted from https://github.com/google/jax-cfd/blob/main/notebooks/demo.ipynb
    """
    # Define the physical dimensions of the simulation.
    grid = cfd.grids.Grid(shape=(size, size),
                          domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

    # Construct a random initial velocity. The `filtered_velocity_field`
    # function ensures that the initial velocity is divergence free and it
    # filters out high frequency fluctuations.
    v0 = cfd.initial_conditions.filtered_velocity_field(
        jax.random.PRNGKey(seed), grid, max_velocity)

    # Define a step function and use it to compute a trajectory.
    step_fn = cfd.funcutils.repeated(
        cfd.equations.semi_implicit_navier_stokes(
            density=density, viscosity=viscosity, dt=dt, grid=grid),
        steps=inner_steps)
    rollout_fn = jax.jit(cfd.funcutils.trajectory(step_fn, outer_steps))
    _, trajectory = jax.device_get(rollout_fn(v0))

    return trajectory[0].data, trajectory[1].data
