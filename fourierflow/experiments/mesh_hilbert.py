import enum

import torch
from einops import repeat

from fourierflow.modules.normalizer import Normalizer
from fourierflow.registries import Experiment, Module


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


@Experiment.register('mesh_hilbert')
class MeshHilbert(Experiment):
    def __init__(self,
                 conv: Module,
                 output_dim: int = 2,
                 warmup_steps: int = 1000,
                 max_accumulations: int = int(1e6),
                 predict_diff: bool = True,
                 clip_val: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.conv = conv
        if max_accumulations > 0:
            self.output_normalizer = Normalizer(
                [output_dim], max_accumulations)
        self.output_dim = output_dim
        self.predict_diff = predict_diff

        self.automatic_optimization = False  # activates manual optimization
        self.clip_val = clip_val
        self.warmup_steps = warmup_steps

    def forward(self, x):
        return x

    def get_loss(self, batch, preds):
        # Build target velocity change
        cur_velocity = batch['velocity'][0]
        targets = batch['target_velocity'][0]
        if self.predict_diff:
            targets = targets - cur_velocity
        if hasattr(self, 'output_normalizer'):
            targets = self.output_normalizer(targets)

        # Build loss
        node_type = batch['node_type'][0]
        loss_mask = torch.logical_or(node_type == NodeType.NORMAL,
                                     node_type == NodeType.OUTFLOW)
        T = preds.shape[0]
        loss_mask = repeat(loss_mask, 'n -> t n', t=T)
        error = torch.sum((targets - preds)**2, dim=2)
        loss = torch.mean(error[loss_mask])

        return loss

    def learning_step(self, batch):
        velocity = batch['velocity'][0]
        # velocity.shape == [batch_size, n_nodes, 2]

        mesh_pos = batch['mesh_pos'][0]
        # mesh_pos.shape == [n_nodes, 2]

        T = velocity.shape[0]
        mesh_pos = repeat(mesh_pos, 'n d -> t n d', t=T)
        # mesh_pos.shape == [batch_size, n_nodes, 2]

        feats = torch.cat([velocity, mesh_pos], dim=-1)
        # mesh_pos.shape == [batch_size, n_nodes, 4]

        preds, _, _ = self.conv(feats)

        return preds

    def training_step(self, batch, batch_idx):
        preds = self.learning_step(batch)
        loss = self.get_loss(batch, preds)
        self.log('train_loss', loss, prog_bar=True)

        # Accumulate normalization statistics in the first few steps.
        if self.global_step < self.warmup_steps:
            return

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
        preds = self.learning_step(batch)
        loss = self.get_loss(batch, preds)
        self.log('valid_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        preds = self.learning_step(batch)
        loss = self.get_loss(batch, preds)
        self.log('test_loss', loss)
