
import logging
from pathlib import Path
from typing import List, Optional

import jax
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.base import DummyLogger
from tqdm import tqdm

from fourierflow.callbacks import Callback

from .jax_callback_hook import TrainerCallbackHookMixin

logger = logging.getLogger(__name__)


class JAXTrainer(TrainerCallbackHookMixin):
    def __init__(
        self,
        max_epochs,
        weights_save_path: Optional[str] = None,
        enable_model_summary: bool = False,
        resume_from_checkpoint=None,
        limit_train_batches=None,
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[WandbLogger] = None,
        seed: Optional[int] = None,
        enable_checkpointing: bool = True,
        plugins=None,
    ):
        self.max_epochs = max_epochs
        self.weights_save_path = weights_save_path
        if weights_save_path:
            self.weights_save_path = Path(weights_save_path)
        self.limit_train_batches = limit_train_batches
        self.callbacks = callbacks or []
        self.current_epoch = -1
        self.routine = None
        self.seed = seed
        self.logger = logger or DummyLogger()
        self.global_step = -1
        self.logs = {}

    def tune(self, routine, datamodule):
        pass

    def fit(self, routine, datamodule):
        self.routine = routine
        params = routine.init(self.seed)
        opt_state = routine.optimizer.init(params)
        step = jax.jit(routine.step)

        self.on_train_start()
        for epoch in range(self.max_epochs):
            self.current_epoch += 1
            self.on_train_epoch_start()
            train_batches = iter(datamodule.train_dataloader())
            with tqdm(train_batches, total=self.limit_train_batches, unit="batch") as tepoch:
                for i, batch in enumerate(tepoch):
                    self.global_step += 1
                    self.on_train_batch_start(batch, i)
                    tepoch.set_description(f"Epoch {epoch}")
                    outputs = step(params, opt_state, batch)
                    params, opt_state, loss_value = outputs
                    routine.params = params
                    tepoch.set_postfix(loss=loss_value.item())
                    self.on_train_batch_end(outputs, batch, i)

                    if self.global_step % 100 == 0:
                        logs = {'loss': loss_value.item()}
                        self.logger.log_metrics(logs, step=self.global_step)

                    if self.limit_train_batches and i >= self.limit_train_batches:
                        break
            self.on_train_epoch_start()

            self.on_validation_epoch_start()
            validate_batches = iter(datamodule.val_dataloader())
            for i, batch in tqdm(enumerate(validate_batches)):
                assert i == 0, "Need to implement log merging first"
                self.on_validation_batch_start(batch, i, 0)
                outputs = routine.valid_step(params, **batch)
                logs = outputs
                logs = {f'valid_{k}': v for k, v in logs.items()}
                self.logs = logs
                scalars = {k: v for k, v in logs.items() if np.isscalar(v)}
                # TODO: merge logs when there is more than one batch
                self.logger.log_metrics(scalars, step=self.global_step)
                self.on_validation_batch_end(outputs, batch, i, 0)
            self.on_validation_epoch_end()

        self.on_train_end()

    def test(self, routine, datamodule):
        self.on_test_epoch_start()
        test_batches = iter(datamodule.test_dataloader())
        for i, batch in tqdm(enumerate(test_batches)):
            assert i == 0, "Need to implement log merging first"
            self.on_test_batch_start(batch, i, 0)
            outputs = routine.valid_step(routine.params, **batch)
            logs = outputs
            logs = {f'test_{k}': v for k, v in logs.items()}
            # TODO: merge logs when there is more than one batch
            scalars = {k: v for k, v in logs.items() if np.isscalar(v)}
            self.logger.log_metrics(scalars)

            if 'test_correlations' in logs:
                corr_rows = list(zip(logs['test_times'],
                                     logs['test_correlations']))
                self.logger.experiment.log({
                    'test_correlations': wandb.Table(['time', 'corr'], corr_rows)
                })

            self.on_test_batch_end(outputs, batch, i, 0)
        self.on_test_epoch_end()
