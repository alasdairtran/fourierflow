import enum
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common import Lazy
from allennlp.training.optimizers import Optimizer

from fourierflow.builders import NodeType
from fourierflow.modules.normalizer import Normalizer
from fourierflow.registries import Experiment, Module, Scheduler


@Experiment.register('mesh_fourier')
class MeshFourierExperiment(Experiment):
    def __init__(self,
                 model: Module,
                 warmup_steps: int,
                 max_accumulations: int,
                 clip_val: float,
                 noise_std: float,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.noise_std = noise_std

        self.input_size = model.input_size
        self.output_size = model.output_size
        self.input_normalizer = Normalizer([self.input_size],
                                           max_accumulations)
        self.output_normalizer = Normalizer([self.output_size],
                                            max_accumulations)

        self.automatic_optimization = False  # activates manual optimization
        self.clip_val = clip_val
        self.warmup_steps = warmup_steps

    def get_loss(self, batch, targets, preds):
        node_type = batch['node_type'][0]
        loss_mask = torch.logical_or(node_type == NodeType.NORMAL,
                                     node_type == NodeType.OUTFLOW)
        error = torch.sum((targets - preds)**2, dim=1)
        loss = torch.mean(error[loss_mask])

        return loss

    def preprocess_batch(self, batch):
        # Each node has a type: 0 (normal), 4 (inflow), 5 (outflow), 6 (wall)
        node_types = batch['node_type'][0]
        # node_types.shape == [n_nodes]

        node_types = F.one_hot(node_types.long(), num_classes=7)
        # node_types.shape == [n_nodes, 7]

        velocities = batch['velocity'][0]
        # velocities.shape == [n_nodes, 2]

        mesh_pos = batch['mesh_pos'][0]
        # mesh_pos.shape == [n_nodes, 2]

        node_features = torch.cat([velocities, node_types, mesh_pos], dim=-1)
        # node_features.shape == [n_nodes, 11]

        inputs = self.input_normalizer(node_features)
        noise = torch.randn(*inputs.shape, device=inputs.device)
        inputs += noise * self.noise_std

        # Build target velocity change
        cur_velocity = batch['velocity'][0]
        target_velocity = batch['target_velocity'][0]
        target_velocity_change = target_velocity - cur_velocity
        targets = self.output_normalizer(target_velocity_change)

        return inputs, targets

    def forward(self, batch, inputs):
        mass = batch['mass'][0]
        # mass.shape == [n_nodes, n_nodes]

        basis = batch['basis'][0]
        # basis.shape == [n_nodes, n_basis]

        preds = self.model(inputs, mass, basis)['preds']

        return preds

    def manual_optimize(self, loss):
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        for group in opt.param_groups:
            torch.nn.utils.clip_grad_value_(group["params"], self.clip_val)
        opt.step()

        sch = self.lr_schedulers()
        sch.step()

    def training_step(self, batch, batch_idx):
        # Accumulate normalization statistics in the first few steps.
        inputs, targets = self.preprocess_batch(batch)
        self._log_norm_stats()
        if self.global_step < self.warmup_steps:
            return

        preds = self.forward(batch, inputs)
        loss = self.get_loss(batch, targets, preds)
        self.log('train_loss', loss,  prog_bar=True)

        self.manual_optimize(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = self.preprocess_batch(batch)
        preds = self.forward(batch, inputs)
        loss = self.get_loss(batch, targets, preds)
        self.log('valid_loss', loss,  prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = self.preprocess_batch(batch)
        preds = self.forward(batch, inputs)
        loss = self.get_loss(batch, targets, preds)
        self.log('test_loss', loss)

    def _log_norm_stats(self):
        for i in range(self.input_size):
            self.log(f'input_norm_mean_{i}', self.input_normalizer.mean[i])
            self.log(f'input_norm_std_{i}', self.input_normalizer.std[i])

        for i in range(self.output_size):
            self.log(f'output_norm_mean_{i}', self.output_normalizer.mean[i])
            self.log(f'output_norm_std_{i}', self.output_normalizer.std[i])
