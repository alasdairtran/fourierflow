from functools import partial
from pathlib import Path
from typing import List, Optional

import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import ptvsd
import xarray as xr
from jax_cfd.base.boundaries import periodic_boundary_conditions
from jax_cfd.base.funcutils import init_context
from jax_cfd.base.grids import Grid, GridArray, GridVariable
from jax_cfd.ml.advections import modular_self_advection, self_advection
from jax_cfd.ml.equations import modular_navier_stokes_model
from jax_cfd.ml.forcings import kolmogorov_forcing
from jax_cfd.ml.interpolations import FusedLearnedInterpolation
from jax_cfd.ml.physics_specifications import NavierStokesPhysicsSpecs
from omegaconf import OmegaConf
from typer import Argument, Typer

app = Typer()


@app.callback(invoke_without_command=True)
def main(config_dir: str,
         overrides: Optional[List[str]] = Argument(None),
         debug: bool = False):
    """Train a JAX experiment."""
    hydra.initialize(config_path=Path('../..') / config_dir)
    config = hydra.compose(config_name='config', overrides=overrides)
    OmegaConf.set_struct(config, False)

    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        ptvsd.enable_attach(address=('0.0.0.0', 5678))
        ptvsd.wait_for_attach()

    # Load data
    ds = xr.open_dataset(config.builder.train_path, engine='h5netcdf')
    vorticities = ds.vorticity.values

    grid = Grid(shape=(2048, 2048), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    dt = 0.001
    forcing_module = kolmogorov_forcing(
        grid, scale=1.0, wavenumber=4, linear_coefficient=-0.1)
    physics_specs = NavierStokesPhysicsSpecs(
        density=1.0,
        viscosity=1e-3,
        forcing_module=forcing_module)

    interpolation_module = partial(FusedLearnedInterpolation, tags=('u', 'c'))
    advection_module = partial(
        modular_self_advection, interpolation_module=interpolation_module)
    convection_module = partial(
        self_advection, advection_module=advection_module)

    def step_fwd(x):
        model = modular_navier_stokes_model(
            grid, dt, physics_specs,
            convection_module=convection_module)
        return model(x)

    step_model = hk.without_apply_rng(hk.transform(step_fwd))

    inputs = []
    for seed, offset in enumerate(grid.cell_faces):
        rng_key = jax.random.PRNGKey(seed)
        data = jax.random.uniform(rng_key, grid.shape, jnp.float32)
        variable = GridVariable(
            array=GridArray(data, offset, grid),
            bc=periodic_boundary_conditions(grid.ndim))
        inputs.append(variable)
    inputs = tuple(inputs)
    rng = jax.random.PRNGKey(42)

    with init_context():
        params = step_model.init(rng, inputs)

    u1, v1 = step_model.apply(params, inputs)


if __name__ == "__main__":
    app()
