import torch
import torch.nn.functional as F
from rivernet.modules.fourier import SimpleBlock1d
from rivernet.modules.loss import LpLoss

from .base import System


@System.register('fourier_1d')
class Fourier1D(System):
    def __init__(self, modes, width):
        super().__init__()
        self.conv = SimpleBlock1d(modes, width)
        self.l2_loss = LpLoss(size_average=True)

    def forward(self, x):
        x = self.conv(x)
        return x.squeeze()

    def training_step(self, batch, batch_idx):
        X, y = batch
        B = X.shape[0]

        out = self.forward(X)
        mse = F.mse_loss(out, y, reduction='mean')
        l2_loss = self.l2_loss(out.view(B, -1), y.view(B, -1))

        self.log('train_mse', mse)
        self.log('train_l2_loss', l2_loss)

        return l2_loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        B = X.shape[0]

        out = self.forward(X)
        mse = F.mse_loss(out, y, reduction='mean')
        l2_loss = self.l2_loss(out.view(B, -1), y.view(B, -1))

        self.log('valid_mse', mse)
        self.log('valid_l2_loss', l2_loss)

        return l2_loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=100, gamma=0.5)
        return [opt], [scheduler]
