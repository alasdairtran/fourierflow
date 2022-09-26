from typing import Optional

import torch
from torch import nn

from fourierflow.modules.loss import LpLoss

from .base import Routine


class PointCloudExperiment(Routine):
    def __init__(self,
                 model: nn.Module,
                 iphi: nn.Module,
                 N: int,
                 automatic_optimization: bool = True,
                 accumulate_grad_batches: int = 1,
                 clip_val: Optional[float] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.iphi = iphi
        self.N = N
        self.l2_loss = LpLoss(size_average=True)
        self.automatic_optimization = automatic_optimization
        self.accumulate_grad_batches = accumulate_grad_batches
        self.clip_val = clip_val

    def forward(self, data):
        return

    def training_step(self, batch, batch_idx):
        xy, rr, sigma = batch['xy'], batch['rr'], batch['sigma']
        B = rr.shape[0]

        out = self.model(xy, code=rr, iphi=self.iphi)
        loss_data = self.l2_loss(out.view(B, -1), sigma.view(B, -1))

        samples_x = torch.rand(B, self.N, 2).cuda() * 3 -1
        samples_xi = self.iphi(samples_x, code=rr)
        loss_reg = self.l2_loss(samples_xi, samples_x)
        loss = loss_data + 0 * loss_reg

        self.log('train_loss', loss)
        self.log('train_loss_reg', loss_reg)

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

        return loss

    def validation_step(self, batch, batch_idx):
        xy, rr, sigma = batch['xy'], batch['rr'], batch['sigma']
        B = rr.shape[0]

        out = self.model(xy, code=rr, iphi=self.iphi)
        loss = self.l2_loss(out.view(B, -1), sigma.view(B, -1))

        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        xy, rr, sigma = batch['xy'], batch['rr'], batch['sigma']
        B = rr.shape[0]

        out = self.model(xy, code=rr, iphi=self.iphi)
        loss = self.l2_loss(out.view(B, -1), sigma.view(B, -1))

        self.log('test_loss', loss)
        return loss
