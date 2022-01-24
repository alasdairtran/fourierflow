from typing import Any, Mapping, Tuple

import elegy as eg
import haiku as hk
import jax
import jax.numpy as jnp
from elegy.model.model import M
from elegy.model.model_core import (PredStepOutput, TestStepOutput,
                                    TrainStepOutput)
from elegy.types import Logs
from jax_cfd.base.boundaries import periodic_boundary_conditions
from jax_cfd.base.funcutils import init_context
from jax_cfd.base.grids import Grid, GridArray, GridVariable
from jax_cfd.ml.advections import ConvectionModule
from jax_cfd.ml.equations import modular_navier_stokes_model
from jax_cfd.ml.physics_specifications import BasePhysicsSpecs
from optax import l2_loss
from treex import Inputs, LossAndLogs


class LearnedInterpolator(eg.Model):
    def __init__(self,
                 size: int,
                 dt: float,
                 physics_specs: BasePhysicsSpecs,
                 convection_module: ConvectionModule,
                 **kwargs):
        grid = Grid(shape=(size, size),
                    domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

        def step_fwd(vx, vy, vorticity):
            inputs = []
            for v, offset in zip([vx, vy], grid.cell_faces):
                variable = GridVariable(
                    array=GridArray(v[0], offset, grid),
                    bc=periodic_boundary_conditions(grid.ndim))
                inputs.append(variable)
            model = modular_navier_stokes_model(
                grid, dt, physics_specs,
                convection_module=convection_module)
            return model(inputs)

        step_model = hk.transform_with_state(step_fwd)

        super().__init__(module=step_model, **kwargs)

    def init_step(self: M, key: jnp.ndarray, inputs: Any) -> M:
        model: M = self

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

        preds, model = model.pred_step(inputs)
        vx_pred, vy_pred = preds[0].array.data, preds[1].array.data
        vx_loss = l2_loss(vx_pred, inputs['vx'][0]).mean()
        vy_loss = l2_loss(vy_pred, inputs['vy'][0]).mean()
        loss = vx_loss + vy_loss

        logs = {'loss': loss}

        return loss, logs, model

    def test_step(self: M, inputs: Any, labels: Mapping[str, Any]) -> TestStepOutput[M]:
        model: M = self

        T = inputs['data'].shape[-1]  # [1, 32, 32, 2441]

        for t in range(T):
            batch = {
                'vorticity': inputs['data'][..., t],
                'vx': inputs['vx'][..., t],
                'vy': inputs['vy'][..., t],
            }
            preds, model = model.pred_step(batch)
            vx_pred, vy_pred = preds[0].array.data, preds[1].array.data
            vx_loss = l2_loss(vx_pred, batch['vx'][0]).mean()
            vy_loss = l2_loss(vy_pred, batch['vy'][0]).mean()
            loss = vx_loss + vy_loss

        logs = {'loss': loss}

        return loss, logs, model

    def pred_step(self: M, inputs: Any) -> PredStepOutput[M]:
        model: M = self
        inputs_obj = Inputs.from_value(inputs)

        preds = model.module(*inputs_obj.args, **inputs_obj.kwargs)

        return preds, model
