import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import dask
import dask.array as da
import h5py
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import ptvsd
import torch
import xarray as xr
from dask.delayed import Delayed
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from hydra.utils import instantiate
from jax_cfd.base.equations import stable_time_step
from jax_cfd.base.grids import Grid
from omegaconf import OmegaConf
from typer import Argument, Option, Typer

from fourierflow.builders import generate_kolmogorov
from fourierflow.builders.synthetic import (Force, GaussianRF,
                                            solve_navier_stokes_2d)

logger = logging.getLogger(__name__)

app = Typer()


@app.command()
def kolmogorov(
        config_path: Path,
        devices: str = Option('0', help='Comma-separated list of GPUs'),
        overrides: Optional[List[str]] = Argument(None),
        refresh: float = Option(1, help='How often to refresh progress bar'),
        debug: bool = Option(False, help='Enable debugging mode with ptvsd')):
    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        ptvsd.enable_attach(address=('0.0.0.0', 5678))
        ptvsd.wait_for_attach()

    device_list = [int(d) for d in devices.split(',')]
    if len(device_list) > 1:
        cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=device_list)
        client = Client(cluster)

    config_dir = config_path.parent
    stem = config_path.stem
    hydra.initialize(config_path=str('../..' / config_dir))
    c = hydra.compose(config_name=stem, overrides=overrides or [])
    OmegaConf.set_struct(c, False)

    # Define the physical dimensions of the simulation.
    sim_grid = instantiate(c.sim_grid)
    out_grids = {}
    for size in c.out_sizes:
        grid = Grid(shape=(size, size), domain=sim_grid.domain)
        out_grids[size] = grid

    # Automatically determine the time step if not specified in config.
    dt = stable_time_step(c.max_velocity, c.cfl_safety_factor,
                          c.equation.kwargs.viscosity, sim_grid)
    dt = c.get('time_step', dt)

    rng_key = jax.random.PRNGKey(c.seed)
    keys = jax.random.split(rng_key, c.n_trajectories)

    init_path = c.get('init_path', None)
    if init_path:
        init_ds = xr.open_dataset(c.init_path, engine='h5netcdf')
        vorticities0 = init_ds.vorticity.values
        assert vorticities0.shape[1] == sim_grid.shape[0]

    if c.outer_steps > 0:
        shapes = {size: (c.outer_steps, size, size) for size in c.out_sizes}
        dim_names = ('sample', 'time', 'x', 'y')
        coords = {size: {
            'sample': range(c.n_trajectories),
            'time': dt * c.inner_steps * np.arange(1, c.outer_steps + 1),
            'x': out_grids[size].axes()[0],
            'y': out_grids[size].axes()[1],
        } for size in c.out_sizes}
    else:
        shapes = {size: (size, size) for size in c.out_sizes}
        dim_names = ('sample', 'x', 'y')
        coords = {size: {
            'sample': range(c.n_trajectories),
            'x': out_grids[size].axes()[0],
            'y': out_grids[size].axes()[1],
        } for size in c.out_sizes}

    # Appending to netCDF files is not supported yet, but we can use
    # dask.delayed to save simulations in a streaming fashion. See:
    # https://stackoverflow.com/a/46958947/3790116
    # https://github.com/pydata/xarray/issues/1672

    gvars: Dict[int, Dict] = {size: {
        'vxs': [], 'vys': [], 'vorticities': []
    } for size in c.out_sizes}
    durations = []

    for i in range(c.n_trajectories):
        outs = dask.delayed(generate_kolmogorov)(
            sim_grid=sim_grid,
            out_sizes=c.out_sizes,
            dt=dt,
            equation=c.equation,
            peak_wavenumber=c.peak_wavenumber,
            max_velocity=c.max_velocity,
            seed=keys[i],
            vorticity0=vorticities0[i] if init_path else None,
            inner_steps=c.inner_steps,
            outer_steps=c.outer_steps,
            warmup_steps=c.warmup_steps)
        trajs, elapsed = outs[0], outs[1]

        for size in c.out_sizes:
            shape = shapes[size]
            traj = trajs[size]
            vx = da.from_delayed(traj['vx'], shape, np.float32)
            vy = da.from_delayed(traj['vy'], shape, np.float32)
            vorticity = da.from_delayed(traj['vorticity'], shape, np.float32)
            gvars[size]['vxs'].append(vx)
            gvars[size]['vys'].append(vy)
            gvars[size]['vorticities'].append(vorticity)

        durations.append(da.from_delayed(elapsed, (), np.float32))

    for size in c.out_sizes:
        gvars[size]['vxs'] = da.stack(gvars[size]['vxs'])
        gvars[size]['vys'] = da.stack(gvars[size]['vys'])
        gvars[size]['vorticities'] = da.stack(gvars[size]['vorticities'])
    durations = da.stack(durations)

    attrs = pd.json_normalize(OmegaConf.to_object(c), sep='.')
    attrs = attrs.to_dict(orient='records')[0]
    attrs = {k: (str(v) if isinstance(v, bool) else v)
             for k, v in attrs.items()}

    ds_dict = {}
    for size in c.out_sizes:
        ds_dict[size] = xr.Dataset(
            data_vars={
                'vx': (dim_names, gvars[size]['vxs']),
                'vy': (dim_names, gvars[size]['vys']),
                'vorticity': (dim_names, gvars[size]['vorticities']),
                'elapsed': ('sample', durations),
            },
            coords=coords[size],
            attrs={
                **attrs,
                'dt': dt,
                'gpu': jax.devices()[0].device_kind,
            }
        )

    tasks = []
    for size, ds in ds_dict.items():
        path = config_dir / f'{stem}_{size}.nc'
        task: Delayed = ds.to_netcdf(path, engine='h5netcdf', compute=False)
        tasks.append(task)

    if len(device_list) > 1:
        dask.compute(tasks)
    else:
        with ProgressBar(dt=refresh):
            dask.compute(tasks, num_workers=1)


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
