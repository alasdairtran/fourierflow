import torch
import torch.nn.functional as F
from rivernet.modules import Module

from .base import System
from .viz import plot_deterministic_forecasts


@System.register('nbeats_forecaster')
class NBEATSForecaster(System):
    def __init__(self, model: Module, forecast_len, backcast_len, n_plots):
        super().__init__()
        self.model = model
        self.forecast_len = forecast_len
        self.backcast_len = backcast_len
        self.n_plots = n_plots

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, series, _ = batch
        S, L = self.backcast_len, self.forecast_len
        T = S + L
        series = series.squeeze(-1)[:, :T]
        # series.shape == [batch_size, total_len]

        X = series[:, :S]
        y = series[:, -L:]

        _, preds = self.model(X)
        mse = F.mse_loss(preds, y, reduction='mean')
        self.log('train_mse', mse)

        return mse

    def validation_step(self, batch, batch_idx):
        t, series, mu = batch
        S, L = self.backcast_len, self.forecast_len
        T = S + L
        series = series.squeeze(-1)[:, :T]
        # series.shape == [batch_size, total_len]

        X = series[:, :S]
        y = series[:, -L:]

        _, preds = self.model(X)
        mse = F.mse_loss(preds, y, reduction='mean')
        self.log('train_mse', mse)

        if batch_idx == 0:
            for i in range(self.n_plots):
                plot_deterministic_forecasts(
                    self.logger.experiment, i, t[i], mu[i], t[i, :S],
                    X[i], t[i, -L:], y[i], preds[i])

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters())
        return opt
