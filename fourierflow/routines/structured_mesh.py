from typing import Optional

import torch
from torch import nn

from fourierflow.modules.loss import LpLoss

from .base import Routine


class StructuredMeshExperiment(Routine):
    def __init__(self,
                 model: nn.Module,
                 automatic_optimization: bool = True,
                 accumulate_grad_batches: int = 1,
                 clip_val: Optional[float] = None,
                 loss_scale: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.l2_loss = LpLoss(size_average=True)
        self.automatic_optimization = automatic_optimization
        self.accumulate_grad_batches = accumulate_grad_batches
        self.clip_val = clip_val
        self.loss_scale = loss_scale

    def forward(self, data):
        return

    def training_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        B = x.shape[0]

        out = self.model(x)
        loss = self.l2_loss(out.view(B, -1), y.view(B, -1))
        self.log('train_loss', loss)

        if not self.automatic_optimization:
            if self.accumulate_grad_batches == 1:
                opt = self.optimizers()
                opt.zero_grad()
                self.manual_backward(loss)
                if self.clip_val:
                    for group in opt.param_groups:
                        torch.nn.utils.clip_grad_norm_(group["params"],
                                                        self.clip_val)
                opt.step()

            else:
                opt = self.optimizers()
                loss /= self.accumulate_grad_batches
                self.manual_backward(loss)
                if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                    if self.clip_val:
                        for group in opt.param_groups:
                            torch.nn.utils.clip_grad_norm_(group["params"],
                                                            self.clip_val)
                    opt.step()
                    opt.zero_grad()

            sch = self.lr_schedulers()
            sch.step()

        return loss * self.loss_scale

    def validation_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        B = x.shape[0]

        out = self.model(x)
        loss = self.l2_loss(out.view(B, -1), y.view(B, -1))

        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        B = x.shape[0]

        out = self.model(x)
        loss = self.l2_loss(out.view(B, -1), y.view(B, -1))

        self.log('test_loss', loss)
        return loss
