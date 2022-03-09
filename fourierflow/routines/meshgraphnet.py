import enum
import pickle
from dataclasses import dataclass, replace
from typing import Callable, List

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from einops import repeat
from jax.random import PRNGKey, split, uniform


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


@dataclass
class EdgeSet:
    name: str
    features: jnp.ndarray
    senders: jnp.ndarray
    receivers: jnp.ndarray


@dataclass
class MultiGraph:
    node_features: jnp.ndarray
    edge_sets: List[EdgeSet]


def triangles_to_edges(faces):
    """Compute mesh edges from triangles."""
    # faces.shape == int32 [n_faces, 3]

    # Extract the three edges from the each face
    edges_1 = faces[:, 0:2]
    edges_2 = faces[:, 1:3]
    edges_3 = jnp.stack([faces[:, 2], faces[:, 0]], axis=1)
    edges = jnp.concatenate([edges_1, edges_2, edges_3], axis=0)
    # edges.shape == [n_edges, 2] == [3 * n_faces, 2]

    # Sort each edge so that we always go from a larger index to a smaller index
    receivers = edges.min(axis=1)
    senders = edges.max(axis=1)
    sorted_edges = jnp.stack([senders, receivers], axis=1)
    # sorted_edges.shape == [n_edges, 2]

    # Traverse through 0th dim and remove duplicated edges
    unique_edges = jnp.unique(sorted_edges, axis=0,
                              size=edges.shape[0], fill_value=-1)
    # unique_edges.shape == [n_unique_edges, 2]

    # Unpack again
    senders, receivers = unique_edges[:, 0], unique_edges[:, 1]

    # Create two-way connectivity
    sources = jnp.concatenate([senders, receivers], axis=0)
    dests = jnp.concatenate([receivers, senders], axis=0)
    # sources.shape == dests.shape == [2 * n_unique_edges]

    # Note: -1 pads might be scattered around in no particular order.

    return sources, dests


class MLPEncoder(hk.Module):
    def __init__(self, output_sizes, layer_norm=True, name=None):
        super().__init__(name=name)
        self.layers = []

        n_layers = len(output_sizes)
        for i, size in enumerate(output_sizes):
            components = [hk.Linear(size)]
            if i < n_layers - 1:
                components.append(jax.nn.relu)
            self.layers.append(hk.Sequential(components))

        if layer_norm:
            self.norm = hk.LayerNorm(axis=-1,
                                     create_scale=True,
                                     create_offset=True)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        if hasattr(self, 'norm'):
            x = self.norm(x)

        return x


class GraphEncoder(hk.Module):
    def __init__(self, n_edge_sets=1, name=None):
        super().__init__(name=name)
        self.node_encoder = MLPEncoder(output_sizes=[128, 128],
                                       name='node_encoder')
        self.edge_encoders = []
        for i in range(n_edge_sets):
            self.edge_encoders.append(MLPEncoder(output_sizes=[128, 128],
                                                 name=f'edge_encoder_{i}'))

    def __call__(self, graph):
        node_latents = self.node_encoder(graph.node_features)  # (2059, 128)

        new_edge_sets = []
        for i, edge_set in enumerate(graph.edge_sets):
            latent = self.edge_encoders[i](edge_set.features)  # (23304, 128)
            new_edge_sets.append(replace(edge_set, features=latent))

        graph = MultiGraph(node_features=node_latents,
                           edge_sets=new_edge_sets)
        return graph


class GraphNetBlock(hk.Module):
    def __init__(self, n_edge_sets=1, name=None):
        super().__init__(name=name)
        self.edge_updaters = []
        for i in range(n_edge_sets):
            self.edge_updaters.append(MLPEncoder(output_sizes=[128, 128],
                                                 name=f'edge_updater_{i}'))

        self.node_updater = MLPEncoder(output_sizes=[128, 128],
                                       name='node_updater')

    def _update_edges(self, node_feats, edge_set, i):
        sender_feats = jnp.take(node_feats, edge_set.senders, axis=0)
        receiver_feats = jnp.take(node_feats, edge_set.receivers, axis=0)
        feats_list = [sender_feats, receiver_feats, edge_set.features]

        feats = jnp.concatenate(feats_list, axis=-1)
        return self.edge_updaters[i](feats)

    def _update_nodes(self, node_features, edge_sets):
        feats_list = [node_features]
        for edge_set in edge_sets:
            # n_feats = edge_set.features.shape[-1]
            feats = jnp.zeros(node_features.shape)
            # index = repeat(edge_set.receivers, 'n -> n f', f=n_feats)

            # idx = jnp.meshgrid(*(jnp.arange(n)
            #                    for n in feats.shape), sparse=True)
            # idx[0] = edge_set.receivers # Double check: What happens when idx contains -1?
            feats = feats.at[edge_set.receivers].add(edge_set.features)
            feats_list.append(feats)

        feats = jnp.concatenate(feats_list, axis=-1)
        return self.node_updater(feats)

    def __call__(self, graph):
        # Apply edge functions
        new_edge_sets = []
        for i, edge_set in enumerate(graph.edge_sets):
            # FIXME: unique encoder in each iteration
            updated_feats = self._update_edges(
                graph.node_features, edge_set, i)
            new_edge_sets.append(replace(edge_set, features=updated_feats))

        # Apply node function
        new_node_feats = self._update_nodes(graph.node_features, new_edge_sets)

        # Add residual connections
        new_node_feats = new_node_feats + graph.node_features
        new_edge_sets_2 = []
        for es, old_es in zip(new_edge_sets, graph.edge_sets):
            new_es = replace(es, features=es.features + old_es.features)
            new_edge_sets_2.append(new_es)

        graph = MultiGraph(node_features=new_node_feats,
                           edge_sets=new_edge_sets_2)

        return graph


class GraphProcessor(hk.Module):
    def __init__(self, n_message_passing_steps=15, name=None):
        super().__init__(name=name)
        self.graph_encoder = GraphEncoder()

        self.layers = []
        for i in range(n_message_passing_steps):
            self.layers.append(GraphNetBlock(name=f'graph_layer_{i}'))

        self.graph_decoder = MLPEncoder(output_sizes=[128, 2],
                                        layer_norm=False,
                                        name='node_updater')

    def __call__(self, graph):
        graph = self.graph_encoder(graph)
        for layer in self.layers:
            graph = layer(graph)

        out = self.graph_decoder(graph.node_features)
        return out


class MeshGraphNet:
    def __init__(self,  optimizer: Callable, n_layers: int = 15):
        self.n_layers = n_layers
        self.optimizer = optimizer
        self.params = None

        def model(n_cells, n_nodes, velocity, cells, mesh_pos, node_type, target_velocity, pressure):
            processor = GraphProcessor(name='processor')
            graph = self._build_graph(n_cells, n_nodes, velocity, cells, mesh_pos,
                                      node_type, target_velocity, pressure)

            preds = processor(graph)
            return preds

        self.model = hk.without_apply_rng(hk.transform(jax.vmap(model)))

    def step(self, params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(
            self.loss_fn)(params, batch)
        updates, opt_state = self.optimizer.update(
            grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    def _build_graph(self, n_cells, n_nodes, velocity, cells, mesh_pos, node_type, target_velocity, pressure):
        # Each node has a type: 0 (normal), 4 (inflow), 5 (outflow), 6 (wall)

        node_types = jax.nn.one_hot(node_type, 9)
        # node_types.shape == [n_nodes, 9]

        node_features = jnp.concatenate([velocity, node_types], axis=-1)
        # node_features.shape == [n_nodes, 11] nan padded

        senders, receivers = triangles_to_edges(cells)
        # senders.shape == receivers.shape == [n_edges]

        sender_pos = jnp.take(mesh_pos, senders, axis=0)
        receiver_pos = jnp.take(mesh_pos, receivers, axis=0)
        # sender_pos.shape == receiver_pos.shape == [n_edges, 2] nan padded

        rel_pos = sender_pos - receiver_pos
        # rel_mesh_pos.shape == [n_edges, 2] nan padded

        norms = jnp.linalg.norm(rel_pos, axis=-1, keepdims=True)
        # norms.shape == [n_edges, 1] nan padded

        edge_features = jnp.concatenate([rel_pos, norms], axis=-1)
        # edge_features.shape == [n_edges, 3] nan padded

        # edge_features = self.edge_normalizer(edge_features)
        mesh_edges = EdgeSet(name='mesh_edges',
                             features=edge_features,
                             receivers=receivers,
                             senders=senders)

        # node_features = self.node_normalizer(node_features)
        graph = MultiGraph(node_features=node_features,
                           edge_sets=[mesh_edges])

        return graph

    def load_lightning_model_state(self, path, *args, **kwargs):
        with open(path, 'rb') as f:
            params = pickle.load(f)
            self.params = params

    def init(self, seed, batch):
        rng = PRNGKey(seed)
        params = self.model.init(rng, **batch)

        self.params = params
        return params

    def loss_fn(self, params, batch):
        preds = self.model.apply(params, **batch)
        targets = batch['target_velocity']
        loss = optax.l2_loss(targets, preds).sum(axis=-1)
        loss = jnp.nanmean(loss)
        return loss

    def valid_step(self, params, **batch):
        preds = self.model.apply(params, **batch)
        targets = batch['target_velocity']
        batch_size = targets.shape[0]
        loss = optax.l2_loss(targets, preds).sum(axis=-1)
        loss = jnp.nanmean(loss)

        logs = {
            'loss': loss.item(),
            'weight': batch_size,
        }

        return logs

    def validation_epoch_end(self, outputs):
        logs = {}
        total = np.sum([x['weight'] for x in outputs])
        for key in ['loss']:
            values = [x[key] * x['weight'] for x in outputs]
            logs[f'valid_{key}'] = np.sum(values, axis=0) / total

        return logs

    def test_epoch_end(self, outputs):
        logs = {}
        total = np.sum([x['weight'] for x in outputs])
        for key in ['loss']:
            values = [x[key] * x['weight'] for x in outputs]
            logs[f'test_{key}'] = np.sum(values, axis=0) / total

        return logs

    def cuda(self):
        return self

    def convert_data(self, data):
        return data
