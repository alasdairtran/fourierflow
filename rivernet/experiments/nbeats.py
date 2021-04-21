import matplotlib.pyplot as plt
import pl_bolts
import torch
import wandb
from allennlp.training.learning_rate_schedulers import LinearWithWarmup
from einops import repeat

from rivernet.common import Module

from rivernet.common import Experiment


@Experiment.register('nbeats')
class NBEATSExperiment(Experiment):
    def __init__(self, model: Module = None, backcast_length: int = 42,
                 forecast_length: int = 7, copying_previous_day: bool = False, model_path: str = None):
        super().__init__()
        self.model = model
        self.backcast_len = backcast_length
        self.forecast_len = forecast_length
        self.copying_previous_day = copying_previous_day

    def forward(self, x):
        x = self.model(x)
        return x

    def _learning_step(self, batch):
        views, log_views = batch
        # x.shape == [batch_size, seq_len]

        sources = log_views[:, :self.backcast_len]
        targets = views[:, -self.forecast_len:]

        if self.model:
            _, X = self.model(sources)
            preds = torch.exp(X)
        elif self.copying_previous_day:
            v = views[:, self.backcast_len - 1]
            preds = repeat(v, 'b -> b d', d=self.forecast_len)

        numerator = torch.abs(targets - preds)
        denominator = torch.abs(targets) + torch.abs(preds)
        loss = numerator / denominator
        loss[torch.isnan(loss)] = 0
        loss = loss.mean()

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._learning_step(batch)
        self.log('train_loss', loss)
        self.log('train_smape', 200 * loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._learning_step(batch)
        self.log('valid_loss', loss)
        self.log('valid_smape', 200 * loss)

    def test_step(self, batch, batch_idx):
        loss = self._learning_step(batch)
        self.log('test_loss', loss)
        self.log('test_smape', 200 * loss)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
            opt, warmup_epochs=1, max_epochs=20)
        return [opt], [scheduler]
