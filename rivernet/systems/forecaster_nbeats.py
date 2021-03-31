import torch
import torch.nn.functional as F
from rivernet.modules import Module

from .base import System
from .viz import plot_deterministic_forecasts


@System.register('nbeats_forecaster')
class NBEATSForecaster(System):
    def __init__(self, model: Module, n_plots):
        super().__init__()
        self.model = model
        self.n_plots = n_plots

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, _, _, x, _, _, y = batch
        # x.shape == [batch_size, backcast_len]

        _, preds = self.model(x)
        mse = F.mse_loss(preds, y, reduction='mean')
        self.log('train_mse', mse)

        return mse

    def validation_step(self, batch, batch_idx):
        t, mu, t_x, x, _, t_y, y = batch
        # x.shape == [batch_size, backcast_len]

        _, preds = self.model(x)
        mse = F.mse_loss(preds, y, reduction='mean')
        self.log('valid_mse', mse)

        if batch_idx == 0:
            for i in range(self.n_plots):
                e = 0 if self.global_step == 0 else self.current_epoch + 1
                name = f'e{e:02}-s{i:02}'
                plot_deterministic_forecasts(
                    self.logger.experiment, name, t[i], mu[i], t_x[i],
                    x[i], t_y[i], y[i], preds[i])

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters())
        return opt
