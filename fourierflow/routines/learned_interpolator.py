import pickle
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_cfd.base.boundaries import periodic_boundary_conditions
from jax_cfd.base.funcutils import init_context, scan, trajectory
from jax_cfd.base.grids import Grid, GridArray, GridVariable
from jax_cfd.base.resize import downsample_staggered_velocity
from jax_cfd.ml.advections import ConvectionModule
from jax_cfd.ml.equations import modular_navier_stokes_model
from jax_cfd.ml.physics_specifications import BasePhysicsSpecs

from fourierflow.utils import velocity_to_vorticity


class LearnedInterpolator:
    def __init__(self,
                 size: int,
                 dt: float,
                 physics_specs: BasePhysicsSpecs,
                 convection_module: ConvectionModule,
                 inner_steps: int,
                 outer_steps: int,
                 unroll_length: int,
                 optimizer: Callable,
                 **kwargs):
        sim_grid = Grid(shape=(size, size),
                        domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

        out_grid = Grid(shape=(32, 32),
                        domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

        self.size = size
        self.sim_grid = sim_grid
        self.out_grid = out_grid
        self.inner_steps = inner_steps
        self.outer_steps = outer_steps
        self.unroll_length = unroll_length
        self.step_size = dt * inner_steps
        self.optimizer = optimizer
        self.params = None
        self.n_steps = outer_steps

        def step_fwd(vx, vy):
            inputs = []
            for v, offset in zip([vx, vy], sim_grid.cell_faces):
                variable = GridVariable(
                    array=GridArray(v, offset, sim_grid),
                    bc=periodic_boundary_conditions(sim_grid.ndim))
                inputs.append(variable)
            model = modular_navier_stokes_model(
                sim_grid, dt, physics_specs,
                convection_module=convection_module)
            next_state = model(inputs)
            vx, vy = next_state[0].array.data, next_state[1].array.data
            return {'vx': vx, 'vy': vy}

        self.model = hk.without_apply_rng(hk.transform(jax.vmap(step_fwd)))

    def warmup(self):
        with init_context():
            inputs = {}
            for i, k in enumerate(['vx', 'vy']):
                rgn_key = jax.random.PRNGKey(i)
                data = jax.random.uniform(
                    rgn_key, (1, self.size, self.size, 1), jnp.float32)
                inputs[k] = data
            self.infer(inputs)

    def infer(self, data):
        def step_fn(**x): return self.model.apply(self.params, **x)

        outer_step_fn = repeated(step_fn, self.inner_steps)
        trajectory_fn = trajectory(
            outer_step_fn, self.outer_steps, lambda x: x)
        vx = data['vx'][..., 0]
        vy = data['vy'][..., 0]
        _, trajs = trajectory_fn({'vx': vx, 'vy': vy})
        return trajs

    def load_lightning_model_state(self, path, map_location, remove_keys):
        with open(path, 'rb') as f:
            params = pickle.load(f)
            self.params = params

    def init(self, seed):
        with init_context():
            rng = jax.random.PRNGKey(seed)
            inputs = {}
            for i, k in enumerate(['vx', 'vy']):
                rgn_key = jax.random.PRNGKey(i)
                data = jax.random.uniform(
                    rgn_key, (2, self.size, self.size), jnp.float32)
                inputs[k] = data
            params = self.model.init(rng, **inputs)

        self.params = params
        return params

    def loss_fn(self, params, inputs, outputs):
        state = inputs
        vx_preds, vy_preds = [], []
        for _ in range(self.unroll_length):
            state = self.model.apply(params, **state)
            vx_preds.append(state['vx'])
            vy_preds.append(state['vy'])

        vx_preds = jnp.stack(vx_preds, axis=-1)
        vy_preds = jnp.stack(vy_preds, axis=-1)
        vx_loss = optax.l2_loss(vx_preds, outputs['vx']).mean(axis=0).sum()
        vy_loss = optax.l2_loss(vy_preds, outputs['vy']).mean(axis=0).sum()
        loss = vx_loss + vy_loss
        return loss

    def valid_step(self, params, times, vx, vy, targets):
        preds = self.unroll(params, vx, vy)
        # preds.shape == [B, M, N, outer_steps]

        preds_norm = jnp.linalg.norm(preds, axis=(1, 2), keepdims=True)
        targets_norm = jnp.linalg.norm(targets, axis=(1, 2), keepdims=True)
        rho = (preds / preds_norm) * (targets / targets_norm)
        rho = rho.sum(axis=(1, 2)).mean(axis=0)
        rho = jnp.nan_to_num(rho)
        # rho.shape == [outer_steps]

        loss = -rho.mean()

        has_diverged = rho < 0.95
        diverged_idx = has_diverged.nonzero(
            size=1, fill_value=len(has_diverged))
        diverged_t = diverged_idx[0][0]
        time_until = diverged_t * self.step_size

        logs = {
            'loss': loss.item(),
            'time_until': time_until.item(),
            'rho': -loss.item(),
            'correlations': np.array(rho),
            'times': np.array(times[0]),
        }

        return logs

    def unroll(self, params, vx, vy):
        def downsample(velocity):
            vx, vy = velocity['vx'], velocity['vy']
            B, M, N = vx.shape
            vorticities = []

            for b in range(B):
                u, v = vx[b], vy[b]
                if M > 32:
                    vel = (GridArray(u, offset=(1, 0.5), grid=self.sim_grid),
                           GridArray(v, offset=(0.5, 1), grid=self.sim_grid))
                    u, v = downsample_staggered_velocity(
                        self.sim_grid, self.out_grid, vel)
                else:
                    u = GridArray(u, offset=(1, 0.5), grid=self.sim_grid)
                    v = GridArray(v, offset=(0.5, 1), grid=self.sim_grid)
                vort = velocity_to_vorticity(u, v, self.out_grid)
                vorticities.append(vort)

            vorticities = jnp.stack(vorticities, axis=0)
            return vorticities

        def step_fn(**x): return self.model.apply(params, **x)

        outer_step_fn = repeated(step_fn, self.inner_steps)
        trajectory_fn = trajectory(outer_step_fn, self.outer_steps, downsample)
        _, trajs = trajectory_fn({'vx': vx, 'vy': vy})
        trajs = trajs.transpose((1, 2, 3, 0))
        # trajs.shape == [B, M, N, outer_steps]

        return trajs

    def cuda(self):
        return self

    def convert_data(self, data):
        return data


def repeated(f: Callable, steps: int) -> Callable:
    """Returns a repeatedly applied version of f()."""
    def f_repeated(x_initial):
        def g(x, _): return (f(**x), None)
        x_final, _ = scan(g, x_initial, xs=None, length=steps)
        return x_final
    return f_repeated
