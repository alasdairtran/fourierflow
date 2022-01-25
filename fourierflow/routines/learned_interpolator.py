from typing import Any, Callable, Mapping, Tuple

import elegy as eg
import haiku as hk
import jax
import jax.numpy as jnp
from elegy.model.model import M
from elegy.model.model_core import (PredStepOutput, TestStepOutput,
                                    TrainStepOutput)
from elegy.types import Logs
from jax_cfd.base.boundaries import periodic_boundary_conditions
from jax_cfd.base.funcutils import init_context, scan, trajectory
from jax_cfd.base.grids import Grid, GridArray, GridVariable
from jax_cfd.base.resize import downsample_staggered_velocity
from jax_cfd.ml.advections import ConvectionModule
from jax_cfd.ml.equations import modular_navier_stokes_model
from jax_cfd.ml.physics_specifications import BasePhysicsSpecs
from optax import l2_loss
from tqdm import tqdm
from treex import Inputs, LossAndLogs

from fourierflow.utils import velocity_to_vorticity
from fourierflow.utils.array import downsample_vorticity


class LearnedInterpolator(eg.Model):
    def __init__(self,
                 size: int,
                 dt: float,
                 physics_specs: BasePhysicsSpecs,
                 convection_module: ConvectionModule,
                 inner_steps: int,
                 outer_steps: int,
                 **kwargs):
        sim_grid = Grid(shape=(size, size),
                        domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

        out_grid = Grid(shape=(32, 32),
                        domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

        self.sim_grid = sim_grid
        self.out_grid = out_grid
        self.inner_steps = inner_steps
        self.outer_steps = outer_steps
        self.step_size = dt * inner_steps

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
            return vx, vy

        step_model = hk.transform_with_state(jax.vmap(step_fwd))

        super().__init__(module=step_model, **kwargs)

    def init_step(self: M, key: jnp.ndarray, inputs: Any) -> M:
        model: M = self

        inputs = (inputs['vx'], inputs['vy'])

        with init_context():
            model.module = model.module.init(key, inputs=inputs)

        params = model.parameters()
        model.optimizer = model.optimizer.init(params)

        losses, metrics = model._losses_and_metrics.value
        aux_losses = model.loss_logs()
        aux_metrics = model.metric_logs()

        model.loss_and_logs = LossAndLogs(
            losses=losses,
            metrics=metrics,
            aux_losses=aux_losses,
            aux_metrics=aux_metrics,
        )

        model = model.distributed_strategy.handle_post_init(model)

        return model

    def train_step(self: M, inputs: Any, labels: Mapping[str, Any]) -> TrainStepOutput[M]:
        model: M = self
        grads: M
        logs: Logs

        params = model.parameters(self._is_trainable)

        grad_fn = jax.grad(self.loss_fn, has_aux=True)
        grads, (logs, model) = grad_fn(params, model, inputs, labels)

        model, grads = model.distributed_strategy.handle_model_and_grads(
            model, grads)

        params = model.optimizer.update(grads, params)
        model = model.merge(params)

        return logs, model

    @staticmethod
    def loss_fn(params: M, model: M, inputs: Any, labels: Mapping[str, Any]) -> Tuple[jnp.ndarray, Tuple[Logs, M]]:
        model = model.merge(params)
        loss, logs, model = model.get_loss(inputs, labels)
        return loss, (logs, model)

    def get_loss(self, inputs, labels):
        model: M = self

        vx_pred, vy_pred = model.pred_step(inputs)
        vx_loss = l2_loss(vx_pred, inputs['vx']).mean()
        vy_loss = l2_loss(vy_pred, inputs['vy']).mean()
        loss = vx_loss + vy_loss

        logs = {'loss': loss}

        return loss, logs, model

    def test_step(self: M, inputs: Any, labels: Mapping[str, Any]) -> TestStepOutput[M]:
        model = self
        preds = self.unroll(inputs['vx'][..., 0], inputs['vy'][..., 0])
        # preds.shape == [B, M, N, outer_steps]

        s = self.inner_steps
        e = s + self.outer_steps * s
        targets = inputs['corr_data'][..., s:e:s]
        # targets.shape == [B, M, N, outer_steps]

        preds_norm = jnp.linalg.norm(preds, axis=(1, 2), keepdims=True)
        targets_norm = jnp.linalg.norm(targets, axis=(1, 2), keepdims=True)
        rho = (preds / preds_norm) * (targets / targets_norm)
        rho = rho.sum(axis=(1, 2)).mean(axis=0)
        # rho.shape == [outer_steps]

        loss = -rho.mean()

        has_diverged = rho < 0.95
        diverged_idx = has_diverged.nonzero(
            size=1, fill_value=len(has_diverged))
        diverged_t = diverged_idx[0][0]
        time_until = diverged_t * self.step_size

        logs = {'loss': loss, 'time_until': time_until}

        return loss, logs, model

    def unroll(self, vx, vy):
        def downsample(velocity):
            vx, vy = velocity
            B, M, N = vx.shape
            vorticities = []

            for b in range(B):
                if M > 32:
                    vel = (GridArray(vx[b], offset=(1, 0.5), grid=self.sim_grid),
                           GridArray(vy[b], offset=(0.5, 1), grid=self.sim_grid))
                    u, v = downsample_staggered_velocity(
                        self.sim_grid, self.out_grid, vel)
                vort = velocity_to_vorticity(u, v, self.out_grid)
                vorticities.append(vort)

            vorticities = jnp.stack(vorticities, axis=0)
            return vorticities

        outer_step_fn = repeated(self.module, self.inner_steps)
        trajectory_fn = trajectory(outer_step_fn, self.outer_steps, downsample)
        _, trajs = trajectory_fn((vx, vy))
        trajs = trajs.transpose((1, 2, 3, 0))
        # trajs.shape == [B, M, N, outer_steps]

        return trajs

    def pred_step(self: M, inputs: Any) -> PredStepOutput[M]:
        model: M = self
        preds = model.module(inputs['vx'], inputs['vy'])
        return preds


def repeated(f: Callable, steps: int) -> Callable:
    """Returns a repeatedly applied version of f()."""
    def f_repeated(x_initial):
        def g(x, _): return (f(*x), None)
        x_final, _ = scan(g, x_initial, xs=None, length=steps)
        return x_final
    return f_repeated
