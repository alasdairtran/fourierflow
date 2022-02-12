import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import dask
import dask.array as da
import h5py
import hydra
import jax
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
        jax.config.update('jax_disable_jit', True)

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
    for o in c.out_sizes:
        key = (o['size'], o['k'])
        grid = Grid(shape=[o['size']] * sim_grid.ndim, domain=sim_grid.domain)
        out_grids[key] = grid

    if sim_grid.ndim == 2:
        spatial_dims = ['x', 'y']
    elif sim_grid.ndim == 3:
        spatial_dims = ['x', 'y', 'z']

    # Automatically determine the time step if not specified in config.
    if isinstance(c.time_step, float):
        dt = c.time_step
    else:
        dt = instantiate(c.time_step)

    rng_key = jax.random.PRNGKey(c.seed)
    keys = jax.random.split(rng_key, c.n_trajectories)

    init_path = c.get('init_path', None)
    if init_path:
        initial_ds = xr.open_dataset(c.init_path, engine='h5netcdf')
        assert len(initial_ds.x) == sim_grid.shape[0]

    if c.outer_steps > 0:
        shapes = {(o['size'], o['k']): [c.outer_steps // o['k']] + [o['size']] * sim_grid.ndim
                  for o in c.out_sizes}
        dim_names = ['sample', 'time'] + spatial_dims
        coords = {}
        for o in c.out_sizes:
            key = (o['size'], o['k'])
            coords[key] = {
                'sample': range(c.n_trajectories),
                'time': dt * c.inner_steps * o['k'] * np.arange(1, c.outer_steps // o['k'] + 1),
            }
            for i, dim in enumerate(spatial_dims):
                coords[key][dim] = out_grids[key].axes()[i]

    else:
        shapes = {(o['size'], o['k']): [o['size']] * sim_grid.ndim
                  for o in c.out_sizes}
        dim_names = ['sample'] + spatial_dims
        coords = {}
        for o in c.out_sizes:
            key = (o['size'], o['k'])
            coords[key] = {
                'sample': range(c.n_trajectories),
            }
            for i, dim in enumerate(spatial_dims):
                coords[key][dim] = out_grids[key].axes()[i]
    # Appending to netCDF files is not supported yet, but we can use
    # dask.delayed to save simulations in a streaming fashion. See:
    # https://stackoverflow.com/a/46958947/3790116
    # https://github.com/pydata/xarray/issues/1672

    if sim_grid.ndim == 2:
        gvars: Dict[Tuple[int, int], Dict] = {(o['size'], o['k']): {
            'vx': [], 'vy': [], 'vorticity': [],
        } for o in c.out_sizes}
    elif sim_grid.ndim == 3:
        gvars = {(o['size'], o['k']): {
            'vx': [], 'vy': [], 'vz': [],
        } for o in c.out_sizes}
    durations = []

    for i in range(c.n_trajectories):
        outs = dask.delayed(generate_kolmogorov)(
            sim_grid=sim_grid,
            out_sizes=c.out_sizes,
            method=c.method,
            step_fn=c.step_fn,
            downsample_fn=c.downsample_fn,
            peak_wavenumber=c.peak_wavenumber,
            max_velocity=c.max_velocity,
            seed=keys[i],
            initial_field=initial_ds.isel(sample=i) if init_path else None,
            inner_steps=c.inner_steps,
            outer_steps=c.outer_steps,
            warmup_steps=c.warmup_steps)
        trajs, elapsed = outs[0], outs[1]

        for o in c.out_sizes:
            key = (o['size'], o['k'])
            k = o['k']
            shape = shapes[key]
            traj = trajs[key]
            vx = da.from_delayed(traj['vx'][k-1::k], shape, np.float32)
            gvars[key]['vx'].append(vx)

            vy = da.from_delayed(traj['vy'][k-1::k], shape, np.float32)
            gvars[key]['vy'].append(vy)

            if sim_grid.ndim == 2:
                vorticity = da.from_delayed(
                    traj['vorticity'][k-1::k], shape, np.float32)
                gvars[key]['vorticity'].append(vorticity)
            elif sim_grid.ndim == 3:
                vz = da.from_delayed(traj['vz'][k-1::k], shape, np.float32)
                gvars[key]['vz'].append(vz)

        durations.append(da.from_delayed(elapsed, (), np.float32))

    for o in c.out_sizes:
        key = (o['size'], o['k'])
        gvars[key]['vx'] = da.stack(gvars[key]['vx'])
        gvars[key]['vy'] = da.stack(gvars[key]['vy'])
        if sim_grid.ndim == 2:
            gvars[key]['vorticity'] = da.stack(gvars[key]['vorticity'])
        elif sim_grid.ndim == 3:
            gvars[key]['vz'] = da.stack(gvars[key]['vz'])
    durations = da.stack(durations)

    def normalize(x):
        if isinstance(x, bool):
            return str(x)
        elif isinstance(x, Callable):
            return f'{x.__module__}.{x.__name__}'
        elif isinstance(x, list):
            return [normalize(elem) for elem in x]
        elif isinstance(x, dict):
            return str({key: normalize(value) for key, value in x.items()})
        else:
            return x

    attrs = pd.json_normalize(OmegaConf.to_object(c), sep='.')
    attrs = attrs.to_dict(orient='records')[0]
    for k, v in attrs.items():
        attrs[k] = normalize(v)

    ds_dict = {}
    for o in c.out_sizes:
        key = (o['size'], o['k'])
        data_vars = {'elapsed': ('sample', durations)}
        for k, v in gvars[key].items():
            data_vars[k] = (dim_names, v)

        ds_dict[key] = xr.Dataset(
            data_vars=data_vars,
            coords=coords[key],
            attrs={
                **attrs,
                'dt': dt,
                'gpu': jax.devices()[0].device_kind,
            }
        )

    tasks = []
    for key, ds in ds_dict.items():
        if c.outer_steps > 0:
            path = config_dir / f"{stem}_{key[0]}_{key[1]}.nc"
        else:
            path = config_dir / f"{stem}_{key[0]}.nc"
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
