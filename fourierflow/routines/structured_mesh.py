from torch import nn

from fourierflow.modules.loss import LpLoss

from .base import Routine


class StructuredMeshExperiment(Routine):
    def __init__(self,
                 model: nn.Module,
                 loss_scale: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.l2_loss = LpLoss(size_average=True)
        self.loss_scale = loss_scale

    def forward(self, data):
        return

    def training_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        B = x.shape[0]

        out = self.model(x)
        loss = self.l2_loss(out.view(B, -1), y.view(B, -1))
        self.log('train_loss', loss)

        loss = loss * self.loss_scale
        self.optimize_manually(loss, batch_idx)
        return loss

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
