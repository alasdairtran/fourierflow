import matplotlib.pyplot as plt
import pl_bolts
import torch
import wandb
from allennlp.training.learning_rate_schedulers import LinearWithWarmup
from einops import repeat

from rivernet.common import Experiment, Module


@Experiment.register('radflow')
class RadflowExperiment(Experiment):
    def __init__(self, rnn: Module = None, backcast_length: int = 42,
                 forecast_length: int = 7, copying_previous_day: bool = False, model_path: str = None):
        super().__init__()
        self.decoder = rnn
        self.backcast_len = backcast_length
        self.forecast_len = forecast_length
        self.copying_previous_day = copying_previous_day

    def forward(self, x):
        x = self.decoder(x)
        return x

    def _learning_step(self, batch):
        views, log_views = batch
        # x.shape == [batch_size, seq_len]

        sources = log_views[:, :-1]
        targets = views[:, 1:]

        X = self.decoder(sources)
        preds = torch.exp(X)

        numerator = torch.abs(targets - preds)
        denominator = torch.abs(targets) + torch.abs(preds)
        loss = numerator / denominator
        loss[torch.isnan(loss)] = 0
        loss = loss.mean()

        return loss

    def _inference_step(self, batch):
        views, log_views = batch
        sources = log_views[:, :self.backcast_len]
        targets = views[:, -self.forecast_len:]

        pred_list = []
        for i in range(self.forecast_len):
            if self.copying_previous_day:
                pred_list.append(views[:, self.backcast_len - 1])
                continue
            X = self.decoder(sources)
            pred = torch.exp(X[:, -1])
            pred_list.append(pred)
            sources = torch.cat([sources, X[:, -1:]], dim=-1)
        preds = torch.stack(pred_list, dim=-1)

        numerator = torch.abs(targets - preds)
        denominator = torch.abs(targets) + torch.abs(preds)
        loss = numerator / denominator
        loss[torch.isnan(loss)] = 0
        loss = loss.mean()
        return loss * 200

    def training_step(self, batch, batch_idx):
        loss = self._learning_step(batch)
        self.log('train_loss', loss)
        self.log('train_smape', 200 * loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._learning_step(batch)
        smape = self._inference_step(batch)
        self.log('valid_loss', loss)
        self.log('valid_smape', smape)

    def test_step(self, batch, batch_idx):
        loss = self._learning_step(batch)
        smape = self._inference_step(batch)
        self.log('test_loss', loss)
        self.log('test_smape', smape)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
            opt, warmup_epochs=5, max_epochs=100)
        return [opt], [scheduler]
