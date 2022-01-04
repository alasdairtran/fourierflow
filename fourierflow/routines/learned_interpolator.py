import elegy as eg
import haiku as hk
import jax.numpy as jnp
from jax_cfd.base.boundaries import periodic_boundary_conditions
from jax_cfd.base.grids import Grid, GridArray, GridVariable
from jax_cfd.ml.advections import ConvectionModule
from jax_cfd.ml.equations import modular_navier_stokes_model
from jax_cfd.ml.physics_specifications import BasePhysicsSpecs


class LearnedInterpolator(eg.Model):
    def __init__(self,
                 size: int,
                 dt: float,
                 physics_specs: BasePhysicsSpecs,
                 convection_module: ConvectionModule,
                 **kwargs):
        grid = Grid(shape=(size, size),
                    domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

        def step_fwd(inputs, outputs):
            keys = ['vx', 'vy']
            input_vars = []
            for key, offset in zip(keys, grid.cell_faces):
                variable = GridVariable(
                    array=GridArray(inputs[key][0], offset, grid),
                    bc=periodic_boundary_conditions(grid.ndim))
                input_vars.append(variable)
            model = modular_navier_stokes_model(
                grid, dt, physics_specs,
                convection_module=convection_module)
            return model(input_vars)

        step_model = hk.transform_with_state(step_fwd)

        super().__init__(module=step_model, **kwargs)
        self.grid = grid

    def __call__(self):
        pass
