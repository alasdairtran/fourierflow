import torch
import torch.nn.functional as F
from rivernet.modules import Module

from .base import System


@System.register('nbeats_forecaster')
class NBEATSForecaster(System):
    def __init__(self, model: Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, series, _ = batch
        series = series.squeeze(-1)
        # series.shape == [batch_size, total_len]

        X = series[:, :-20]
        y = series[:, -20:]

        _, preds = self.model(X)
        mse = F.mse_loss(preds, y, reduction='mean')
        self.log('train_mse', mse)

        return mse

    def validation_step(self, batch, batch_idx):
        _, series, _ = batch
        series = series.squeeze(-1)
        # series.shape == [batch_size, total_len]

        X = series[:, :-20]
        y = series[:, -20:]

        _, preds = self.model(X)
        mse = F.mse_loss(preds, y, reduction='mean')
        self.log('train_mse', mse)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters())
        return opt
