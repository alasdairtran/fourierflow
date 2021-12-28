import os
from pathlib import Path

import dask
import dask.array as da
import h5py
import jax.numpy as jnp
import jax_cfd.base as cfd
import numpy as np
import ptvsd
import torch
import xarray as xr
from dask.delayed import Delayed
from dask.diagnostics import ProgressBar
from typer import Argument, Option, Typer

from fourierflow.builders import generate_kolmogorov
from fourierflow.builders.synthetic import (Force, GaussianRF,
                                            solve_navier_stokes_2d)

app = Typer()


@app.command()
def kolmogorov(
    path: Path = Argument(..., help='Path to store the generated samples'),
    n_train: int = Option(32, help='Number of train solutions to generate'),
    size: int = Option(2048, help='Size of the domain'),
    density: float = Option(1.0, help='Density of the fluid'),
    viscosity: float = Option(1e-3, help='Viscosity of the fluid'),
    max_velocity: float = Option(2.0, help='Maximum velocity of the fluid'),
    seed: int = Option(0, help='Random seed'),
    inner_steps: int = Option(25, help='Number of steps in the inner loop'),
    outer_steps: int = Option(200, help='Number of steps in the outer loop'),
    cfl_safety_factor: float = Option(0.5, help='CFL safety factor'),
):
    # Create directories to store the simulation data.
    data_dir = path.parent
    data_dir.mkdir(parents=True, exist_ok=True)

    # Define the physical dimensions of the simulation.
    grid = cfd.grids.Grid(shape=(size, size),
                          domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

    # Choose a time step.
    dt = cfd.equations.stable_time_step(
        max_velocity, cfl_safety_factor, viscosity, grid)

    rs = np.random.RandomState(seed)

    # Appending to netCDF files is not supported yet, but we can use
    # dask.delayed to save simulations in a streaming fashion. See:
    # https://stackoverflow.com/a/46958947/3790116
    # https://github.com/pydata/xarray/issues/1672
    us, vs = [], []
    for _ in range(n_train):
        trajectory = dask.delayed(generate_kolmogorov)(
            size=size,
            dt=dt,
            density=density,
            viscosity=viscosity,
            max_velocity=max_velocity,
            seed=rs.randint(int(1e9)),
            inner_steps=inner_steps,
            outer_steps=outer_steps)
        u = da.from_delayed(trajectory[0], (200, size, size), np.float32)
        v = da.from_delayed(trajectory[1], (200, size, size), np.float32)
        us.append(u)
        vs.append(v)

    us = da.stack(us)
    vs = da.stack(vs)

    ds = xr.Dataset(
        data_vars={
            'u': (('sample', 'time', 'x', 'y'), us),
            'v': (('sample', 'time', 'x', 'y'), vs),
        },
        coords={
            'sample': range(n_train),
            'time': dt * inner_steps * np.arange(outer_steps),
            'x': grid.axes()[0],
            'y': grid.axes()[1],
        },
        attrs={
            'dt': dt,
            'density': density,
            'viscosity': viscosity,
            'max_velocity': max_velocity,
            'seed': seed,
        }
    )

    out_path = data_dir / 'train.nc'
    out: Delayed = ds.to_netcdf(out_path, engine='h5netcdf', compute=False)

    with ProgressBar():
        out.compute(num_workers=1)


@app.command()
def navier_stokes(
    path: str = Argument(..., help='Path to store the generated samples'),
    n_train: int = Option(1000, help='Number of train solutions to generate'),
    n_valid: int = Option(200, help='Number of valid solutions to generate'),
    n_test: int = Option(200, help='Number of test solutions to generate'),
    s: int = Option(256, help='Width of the solution grid'),
    t: int = Option(20, help='Final time step'),
    steps: int = Option(20, help='Number of snapshots from solution'),
    mu: float = Option(1e-5, help='Viscoity'),
    mu_min: float = Option(1e-5, help='Minimium viscoity'),
    mu_max: float = Option(1e-5, help='Maximum viscoity'),
    seed: int = Option(23893, help='Seed value for reproducibility'),
    delta: float = Option(1e-4, help='Internal time step for sovler'),
    batch_size: int = Option(50, help='Batch size'),
    force: Force = Option(Force.li, help='Type of forcing function'),
    cycles: int = Option(2, help='Number of cycles in forcing function'),
    scaling: float = Option(0.1, help='Scaling of forcing function'),
    t_scaling: float = Option(0.2, help='Scaling of time variable'),
    varying_force: bool = Option(False, help='Enable time-varying force'),
    debug: bool = Option(False, help='Enable debugging mode with ptvsd'),
):
    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        ptvsd.enable_attach(address=('0.0.0.0', 5678))
        ptvsd.wait_for_attach()

    device = torch.device('cuda')
    torch.manual_seed(seed)
    np.random.seed(seed + 1234)

    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

    data_f = h5py.File(path, 'a')

    def generate_split(n, split):
        print('Generating split:', split)
        data_f.create_dataset(f'{split}/a', (n, s, s), np.float32)
        if varying_force:
            data_f.create_dataset(f'{split}/f', (n, s, s, steps), np.float32)
        else:
            data_f.create_dataset(f'{split}/f', (n, s, s), np.float32)
        data_f.create_dataset(f'{split}/u', (n, s, s, steps), np.float32)
        data_f.create_dataset(f'{split}/mu', (n,), np.float32)
        b = min(n, batch_size)
        c = 0

        with torch.no_grad():
            for j in range(n // b):
                print('batch', j)
                w0 = GRF.sample(b)

                if mu_min != mu_max:
                    mu = np.random.rand(b) * (mu_max - mu_min) + mu_min

                sol, f = solve_navier_stokes_2d(
                    w0, mu, t, delta, steps, cycles,
                    scaling, t_scaling, force, varying_force)
                data_f[f'{split}/a'][c:(c+b), ...] = w0.cpu().numpy()
                data_f[f'{split}/u'][c:(c+b), ...] = sol

                if force == Force.random:
                    data_f[f'{split}/f'][c:(c+b), ...] = f

                data_f[f'{split}/mu'][c:(c+b)] = mu

                c += b

    generate_split(n_train, 'train')
    generate_split(n_valid, 'valid')
    generate_split(n_test, 'test')


if __name__ == "__main__":
    app()
