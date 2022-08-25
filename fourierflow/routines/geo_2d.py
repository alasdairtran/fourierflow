import torch
from torch import nn

from fourierflow.modules.loss import LpLoss

from .base import Routine


class Geo2DExperiment(Routine):
    def __init__(self,
                 model: nn.Module,
                 iphi: nn.Module,
                 N: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.iphi = iphi
        self.N = N
        self.l2_loss = LpLoss(size_average=True)

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
