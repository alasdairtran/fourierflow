import os
import pickle
import shutil
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from .callback import Callback


class JAXModelCheckpoint(Callback):
    def __init__(self, save_dir, monitor, mode):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.dirpath = None

        assert mode in ['min', 'max']
        self.best_metric = float('inf') if mode == 'min' else -float('inf')

    def on_validation_epoch_end(self, trainer, routine):
        self.__resolve_ckpt_dir(trainer)

        metric = trainer.logs[self.monitor]
        stats = f"{self.monitor}={metric:.4f}"
        path = self.dirpath / f'epoch={trainer.current_epoch}-{stats}.ckpt'

        if self.is_better(metric):
            self.best_metric = metric

            for old_path in list(self.dirpath.glob('*.ckpt')):
                old_path.unlink()

            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(routine.params, f)

    def is_better(self, metric):
        if self.mode == 'min':
            return metric < self.best_metric
        else:
            return metric > self.best_metric

    def __resolve_ckpt_dir(self, trainer) -> None:
        if self.dirpath is not None:
            return  # short circuit

        if trainer.logger is not None:
            version = (
                trainer.logger.version
                if isinstance(trainer.logger.version, str)
                else f"version_{trainer.logger.version}"
            )
            ckpt_path = trainer.weights_save_path / "checkpoints" / version
        else:
            ckpt_path = trainer.weights_save_path / "checkpoints"

        self.dirpath = ckpt_path


class CustomModelCheckpoint(ModelCheckpoint):
    def on_pretrain_routine_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        When pretrain routine starts we build the ckpt dir on the fly
        """
        self.__resolve_ckpt_dir(trainer)
        self._save_function = trainer.save_checkpoint
        if self._save_on_train_epoch_end is None:
            # if the user runs validation multiple times per training epoch, we try to save checkpoint after
            # validation instead of on train epoch end
            self._save_on_train_epoch_end = trainer.val_check_interval == 1.0

    def __resolve_ckpt_dir(self, trainer: "pl.Trainer") -> None:
        """
        Determines model checkpoint save directory at runtime. References attributes from the
        trainer's logger to determine where to save checkpoints.
        The base path for saving weights is set in this priority:

        1.  Checkpoint callback's path (if passed in)
        2.  The default_root_dir from trainer if trainer has no logger
        3.  The weights_save_path from trainer, if user provides it
        4.  User provided weights_saved_path

        The base path gets extended with logger name and version (if these are available)
        and subfolder "checkpoints".
        """
        # Todo: required argument `pl_module` is not used
        if self.dirpath is not None:
            return  # short circuit

        if trainer.logger is not None:
            if trainer.weights_save_path != trainer.default_root_dir:
                # the user has changed weights_save_path, it overrides anything
                save_dir = trainer.weights_save_path
            else:
                save_dir = trainer.logger.save_dir or trainer.default_root_dir

            version = (
                trainer.logger.version
                if isinstance(trainer.logger.version, str)
                else f"version_{trainer.logger.version}"
            )
            ckpt_path = os.path.join(save_dir, "checkpoints", version)
        else:
            ckpt_path = os.path.join(trainer.weights_save_path, "checkpoints")

        ckpt_path = trainer.training_type_plugin.broadcast(ckpt_path)

        self.dirpath = ckpt_path

        if not trainer.fast_dev_run and trainer.should_rank_save_checkpoint:
            self._fs.makedirs(self.dirpath, exist_ok=True)
