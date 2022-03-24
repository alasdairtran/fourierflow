# Implementation based on tf.keras.callbacks.py
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/callbacks.py


from typing import Union

import wandb

from .callback import Callback


class WandbLogger(Callback):
    """
    Callback that streams epoch results to tensorboard events folder.
    Supports all values that can be represented as a string,
    including 1D iterables such as `np.ndarray`.
    ```python
    tensorboard_logger = TensorBoard('runs')
    model.fit(X_train, Y_train, callbacks=[tensorboard_logger])
    ```
    """

    def __init__(
        self,
        update_freq: Union[str, int] = "epoch",
    ) -> None:
        """
        Arguments:
            update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
                writes the losses and metrics to TensorBoard after each batch. The same
                applies for `'epoch'`. If using an integer, let's say `1000`, the
                callback will write the metrics and losses to TensorBoard every 1000
                batches. Note that writing too frequently to TensorBoard can slow down
                your training.
        """
        self.keys = None
        self.write_per_batch = True
        try:
            self.update_freq = int(update_freq)
        except ValueError as e:
            self.update_freq = 1
            if update_freq == "batch":
                self.write_per_batch = True
            elif update_freq == "epoch":
                self.write_per_batch = False
            else:
                raise e

        super().__init__()

    def on_train_begin(self, logs=None):
        self.steps = self.params["steps"]
        self.global_step = 0

    def on_train_batch_end(self, batch: int, logs=None):
        logs = logs or {}
        self.global_step = batch + self.current_epoch * (self.steps)
        if self.global_step % self.update_freq == 0:
            if self.keys is None:
                self.keys = logs.keys()
            wandb.log(logs, step=self.global_step)

    def on_epoch_begin(self, epoch: int, logs=None):
        self.current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.keys is None:
            self.keys = logs.keys()
        wandb.log(logs, step=self.global_step)
