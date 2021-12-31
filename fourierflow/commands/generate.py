import os
from pathlib import Path
from typing import List, Optional

import dask
import dask.array as da
import h5py
import hydra
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import numpy as np
import pandas as pd
import ptvsd
import torch
import xarray as xr
from dask.delayed import Delayed
from dask.diagnostics import ProgressBar
from omegaconf import OmegaConf
from typer import Argument, Option, Typer

from fourierflow.builders import generate_kolmogorov
from fourierflow.builders.synthetic import (Force, GaussianRF,
                                            solve_navier_stokes_2d)

app = Typer()


@app.command()
def kolmogorov(config_path: Path,
               overrides: Optional[List[str]] = Argument(None),
               debug: bool = Option(False, help='Enable debugging mode with ptvsd')):
    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        ptvsd.enable_attach(address=('0.0.0.0', 5678))
        ptvsd.wait_for_attach()

    config_dir = config_path.parent
    stem = config_path.stem
    hydra.initialize(config_path=str('../..' / config_dir))
    c = hydra.compose(config_name=stem, overrides=overrides or [])
    OmegaConf.set_struct(c, False)

    # Define the physical dimensions of the simulation.
    grid = cfd.grids.Grid(shape=(c.size, c.size),
                          domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

    # Choose a time step.
    dt = cfd.equations.stable_time_step(
        c.max_velocity, c.cfl_safety_factor, c.equation.kwargs.viscosity, grid)

    rng_key = jax.random.PRNGKey(c.seed)
    keys = jax.random.split(rng_key, c.n_trajectories)

    # Appending to netCDF files is not supported yet, but we can use
    # dask.delayed to save simulations in a streaming fashion. See:
    # https://stackoverflow.com/a/46958947/3790116
    # https://github.com/pydata/xarray/issues/1672
    vorticities = []
    if c.outer_steps > 0:
        shape = (c.outer_steps, c.size, c.size)
    else:
        shape = (c.size, c.size)
    for i in range(c.n_trajectories):
        trajectory = dask.delayed(generate_kolmogorov)(
            size=c.size,
            dt=dt,
            equation=c.equation,
            peak_wavenumber=c.peak_wavenumber,
            max_velocity=c.max_velocity,
            seed=keys[i],
            inner_steps=c.inner_steps,
            outer_steps=c.outer_steps,
            warmup_steps=c.warmup_steps)
        vorticity = da.from_delayed(trajectory, shape, np.float32)
        vorticities.append(vorticity)

    vorticities = da.stack(vorticities)

    attrs = pd.json_normalize(OmegaConf.to_object(c), sep='.')
    attrs = attrs.to_dict(orient='records')[0]
    attrs = {k: (str(v) if isinstance(v, bool) else v)
             for k, v in attrs.items()}

    if c.outer_steps > 0:
        ds = xr.Dataset(
            data_vars={
                'vorticity': (('sample', 'time', 'x', 'y'), vorticities),
            },
            coords={
                'sample': range(c.n_trajectories),
                'time': dt * c.inner_steps * np.arange(c.outer_steps),
                'x': grid.axes()[0],
                'y': grid.axes()[1],
            },
            attrs={
                **attrs,
                'dt': dt,
                'domain_size': 2 * jnp.pi,
            }
        )
    else:
        ds = xr.Dataset(
            data_vars={
                'vorticity': (('sample', 'x', 'y'), vorticities),
            },
            coords={
                'sample': range(c.n_trajectories),
                'x': grid.axes()[0],
                'y': grid.axes()[1],
            },
            attrs={
                **attrs,
                'dt': dt,
                'domain_size': 2 * jnp.pi,
            }
        )

    path = config_dir / f'{stem}.nc'
    task: Delayed = ds.to_netcdf(path, engine='h5netcdf', compute=False)

    with ProgressBar():
        task.compute(num_workers=1)


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
