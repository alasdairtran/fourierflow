import pickle
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
import optax


class MeshGraphNet:
    def __init__(self, optimizer: Callable):
        self.optimizer = optimizer
        self.params = None

        def model(velocity, cells, mesh_pos, node_type, target_velocity, pressure):
            mlp = hk.Sequential([
                hk.Linear(64), jax.nn.relu,
                hk.Linear(2),
            ])
            preds = mlp(velocity)
            return preds

        self.model = hk.without_apply_rng(hk.transform(jax.vmap(model)))

    def step(self, params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(
            self.loss_fn)(params, batch)
        updates, opt_state = self.optimizer.update(
            grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    def _build_graph(self, batch):
        pass

    def load_lightning_model_state(self, path, *args, **kwargs):
        with open(path, 'rb') as f:
            params = pickle.load(f)
            self.params = params

    def init(self, seed):
        rng = jax.random.PRNGKey(seed)
        inputs = {'cells': None, 'mesh_pos': None, 'node_type': None,
                  'target_velocity': None, 'pressure': None}
        rgn_key = jax.random.PRNGKey(42)
        data = jax.random.uniform(rgn_key, (1, 20, 2), jnp.float32)
        inputs['velocity'] = data
        params = self.model.init(rng, **inputs)

        self.params = params
        return params

    def loss_fn(self, params, batch):
        preds = self.model.apply(params, **batch)
        targets = batch['target_velocity']
        loss = optax.l2_loss(targets, preds).mean(axis=0).sum()
        return loss

    def valid_step(self, params, **batch):
        preds = self.model.apply(params, **batch)
        targets = batch['target_velocity']
        loss = optax.l2_loss(targets, preds).mean(axis=0).sum()

        logs = {
            'loss': loss.item(),
        }

        return logs

    def cuda(self):
        return self

    def convert_data(self, data):
        return data
