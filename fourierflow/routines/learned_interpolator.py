from typing import Any, Mapping

import elegy as eg
import haiku as hk
import jax
import jax.numpy as jnp
from elegy.model.model import M
from elegy.model.model_core import TrainStepOutput
from elegy.types import Logs
from jax_cfd.base.boundaries import periodic_boundary_conditions
from jax_cfd.base.funcutils import init_context
from jax_cfd.base.grids import Grid, GridArray, GridVariable
from jax_cfd.ml.advections import ConvectionModule
from jax_cfd.ml.equations import modular_navier_stokes_model
from jax_cfd.ml.physics_specifications import BasePhysicsSpecs
from treex import LossAndLogs


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

    def init_step(self: M,
                  key: jnp.ndarray,
                  inputs: Any) -> M:
        model: M = self

        with init_context():
            model.module = model.module.init(key, inputs=inputs)

        if model.optimizer is not None:
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

    def train_step(self: M,
                   inputs: Any,
                   labels: Mapping[str, Any]) -> TrainStepOutput[M]:
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
