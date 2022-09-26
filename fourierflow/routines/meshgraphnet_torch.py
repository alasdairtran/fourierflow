import enum
from dataclasses import dataclass, replace
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from fourierflow.modules.normalizer import Normalizer

from .base import Routine


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
    features: torch.Tensor
    senders: torch.Tensor
    receivers: torch.Tensor


@dataclass
class MultiGraph:
    node_features: torch.Tensor
    edge_sets: List[EdgeSet]


class MLPEncoder(nn.Module):
    def __init__(self, input_size, output_size=128, latent_size=128, n_layers=3, layer_norm=True):
        super().__init__()
        self._latent_size = latent_size
        self._n_layers = n_layers

        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = input_size if i == 0 else latent_size
            out_dim = output_size if i == n_layers - 1 else latent_size
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity()))
        self.norm = nn.LayerNorm(latent_size) if layer_norm else nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        return x


class GraphEncoder(nn.Module):
    def __init__(self, n_edge_sets=1):
        super().__init__()
        self.node_encoder = MLPEncoder(input_size=11)
        self.edge_encoders = nn.ModuleList([])
        for _ in range(n_edge_sets):
            self.edge_encoders.append(MLPEncoder(input_size=3))

    def forward(self, graph):
        node_latents = self.node_encoder(graph.node_features)

        new_edge_sets = []
        for i, edge_set in enumerate(graph.edge_sets):
            latent = self.edge_encoders[i](edge_set.features)
            new_edge_sets.append(replace(edge_set, features=latent))

        graph = MultiGraph(node_features=node_latents,
                           edge_sets=new_edge_sets)
        return graph


class GraphNetBlock(nn.Module):
    def __init__(self, n_edge_sets=1):
        super().__init__()
        self.edge_updaters = nn.ModuleList([])
        for _ in range(n_edge_sets):
            self.edge_updaters.append(MLPEncoder(input_size=11 + 3 * 1))

        self.node_updater = MLPEncoder(input_size=11 * 2 + 3)

    def _update_edges(self, node_feats, edge_set, i):
        sender_feats = torch.index_select(node_feats, 0, edge_set.senders)
        receiver_feats = torch.index_select(node_feats, 0, edge_set.receivers)
        feats_list = [sender_feats, receiver_feats, edge_set.features]

        feats = torch.cat(feats_list, dim=-1)
        return self.edge_updaters[i](feats)

    def _update_nodes(self, node_features, edge_sets):
        feats_list = [node_features]
        for edge_set in edge_sets:
            n_feats = edge_set.features.shape[-1]
            feats = node_features.new_zeros(*node_features.shape)
            index = repeat(edge_set.receivers, 'n -> n f', f=n_feats)
            feats.scatter_add_(0, index, edge_set.features)
            feats_list.append(feats)

        feats = torch.cat(feats_list, dim=-1)
        return self.node_updater(feats)

    def forward(self, graph):
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


class GraphProcessor(nn.Module):
    def __init__(self, n_message_passing_steps=15):
        super().__init__()
        self.graph_encoder = GraphEncoder()

        self.layers = nn.ModuleList([])
        for _ in range(n_message_passing_steps):
            self.layers.append(GraphNetBlock())

        self.graph_decoder = MLPEncoder(
            input_size=11, output_size=2, layer_norm=False)

    def forward(self, graph):
        latent_graph = self.graph_encoder(graph)
        for layer in self.layers:
            latent_graph = layer(latent_graph)

        out = self.graph_decoder(graph.node_features)
        return out


def triangles_to_edges(faces):
    """Compute mesh edges from triangles."""
    # faces.shape == int [n_faces, 3]

    # Extract the three edges from the each face
    edges_1 = faces[:, 0:2]
    edges_2 = faces[:, 1:3]
    edges_3 = torch.stack([faces[:, 2], faces[:, 0]], axis=1)
    edges = torch.cat([edges_1, edges_2, edges_3], dim=0)
    # edges.shape == [n_edges, 2] == [3 * n_faces, 2]

    # Sort edges so that we always go from a larger index to a smaller index
    receivers = edges.min(dim=1).values
    senders = edges.max(dim=1).values
    sorted_edges = torch.stack([senders, receivers], dim=1)
    # sorted_edges.shape == [n_edges, 2]

    # Traverse through 0th dim and remove duplciated edges
    unique_edges = sorted_edges.unique(dim=0)
    # unique_edges.shape == [n_unique_edges, 2]

    # Unpack again
    senders, receivers = unique_edges.unbind(dim=1)

    # Create two-way connectivity
    sources = torch.cat([senders, receivers], dim=0)
    dests = torch.cat([receivers, senders], dim=0)
    # sources.shape == dests.shape == [2 * n_unique_edges]

    return sources, dests


class MeshCFD(Routine):
    def __init__(self,
                 node_dim: int = 11,
                 edge_dim: int = 3,
                 output_dim: int = 2,
                 warmup_steps: int = 1000,
                 max_accumulations: int = int(1e6),
                 clip_val: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = GraphProcessor()
        self.node_normalizer = Normalizer([node_dim], max_accumulations)
        self.edge_normalizer = Normalizer([edge_dim], max_accumulations)
        self.output_normalizer = Normalizer([output_dim], max_accumulations)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim

        self.automatic_optimization = False  # activates manual optimization
        self.clip_val = clip_val
        self.warmup_steps = warmup_steps

    def _build_graph(self, batch):
        # Each node has a type: 0 (normal), 4 (inflow), 5 (outflow), 6 (wall)
        node_types = batch['node_type'][0]
        # node_types.shape == [n_nodes]

        node_types = F.one_hot(node_types.long(), num_classes=9)
        # node_types.shape == [n_nodes, 9]

        velocities = batch['velocity'][0]
        # velocities.shape == [n_nodes, 2]

        node_features = torch.cat([velocities, node_types], dim=-1)
        # node_features.shape == [n_nodes, 11]

        cells = batch['cells'][0].long()
        # cells.shape == [n_cells, 3]

        senders, receivers = triangles_to_edges(cells)
        # senders.shape == receivers.shape == [n_edges]

        mesh_pos = batch['mesh_pos'][0]
        # mesh_pos.shape == [n_nodes, 2]

        sender_pos = torch.index_select(mesh_pos, 0, senders)
        receiver_pos = torch.index_select(mesh_pos, 0, receivers)
        # sender_pos.shape == receiver_pos.shape == [n_edges, 2]

        rel_pos = sender_pos - receiver_pos
        # rel_mesh_pos.shape == [n_edges, 2]

        norms = torch.linalg.vector_norm(rel_pos, dim=-1, keepdims=True)
        # norms.shape == [n_edges, 1]

        edge_features = torch.cat([rel_pos, norms], dim=-1)
        # edge_features.shape == [n_edges, 3]

        edge_features = self.edge_normalizer(edge_features)
        mesh_edges = EdgeSet(name='mesh_edges',
                             features=edge_features,
                             receivers=receivers,
                             senders=senders)

        node_features = self.node_normalizer(node_features)
        graph = MultiGraph(node_features=node_features,
                           edge_sets=[mesh_edges])

        return graph

    def forward(self, x):
        return x

    def _log_norm_stats(self, x):
        for i in range(self.node_dim):
            self.log(f'node_normalizer_mean_{i}', self.node_normalizer.mean[i])
            self.log(f'node_normalizer_std_{i}', self.node_normalizer.std[i])

        for i in range(self.edge_dim):
            self.log(f'edge_normalizer_mean_{i}', self.edge_normalizer.mean[i])
            self.log(f'edge_normalizer_std_{i}', self.edge_normalizer.std[i])

    def get_loss(self, batch, network_output):
        # Build target velocity change
        cur_velocity = batch['velocity'][0]
        target_velocity = batch['target_velocity'][0]
        target_velocity_change = target_velocity - cur_velocity
        target_normalized = self.output_normalizer(target_velocity_change)

        # Build loss
        node_type = batch['node_type'][0]
        loss_mask = torch.logical_or(node_type == NodeType.NORMAL,
                                     node_type == NodeType.OUTFLOW)
        error = torch.sum((target_normalized - network_output)**2, dim=1)
        loss = torch.mean(error[loss_mask])

        return loss

    def training_step(self, batch, batch_idx):
        # batch['pressure']  # float [n_nodes, 1]

        graph = self._build_graph(batch)

        # Accumulate normalization statistics in the first few steps.
        if self.global_step < self.warmup_steps:
            return

        network_output = self.model(graph)
        loss = self.get_loss(batch, network_output)
        self.log('train_loss', loss)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        for group in opt.param_groups:
            torch.nn.utils.clip_grad_value_(group["params"], self.clip_val)
        opt.step()

        sch = self.lr_schedulers()
        sch.step()

        return loss

    def validation_step(self, batch, batch_idx):
        graph = self._build_graph(batch)
        network_output = self.model(graph)
        loss = self.get_loss(batch, network_output)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        graph = self._build_graph(batch)
        network_output = self.model(graph)
        loss = self.get_loss(batch, network_output)
        self.log('test_loss', loss)
