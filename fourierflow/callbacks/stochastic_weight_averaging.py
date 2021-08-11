# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.trainer.optimizers import _get_default_scheduler_config
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn

from fourierflow.schedulers import SWALR

_AVG_FN = Callable[[torch.Tensor, torch.Tensor,
                    torch.LongTensor], torch.FloatTensor]


class StochasticWeightAveraging(Callback):
    def __init__(self,
                 total_steps: int,
                 swa_step_start: float = 0.75,
                 swa_lr: float = 0.025,
                 annealing_strategy: str = "cos",
                 avg_fn: Optional[_AVG_FN] = None):
        self._total_steps = total_steps
        self._swa_step_start = int(total_steps * swa_step_start)
        rank_zero_info(f'Starting SWA at step {self._swa_step_start}.')
        self._swa_lr = swa_lr
        self._annealing_strategy = annealing_strategy
        self._avg_fn = avg_fn or self.avg_fn
        self._model_contains_batch_norm = None
        self._average_model = None

    @property
    def swa_start(self) -> int:
        return max(self._swa_step_start - 1, 0)  # 0-based

    @property
    def swa_end(self) -> int:
        return self._total_steps  # 0-based

    @staticmethod
    def pl_module_contains_batch_norm(pl_module: "pl.LightningModule"):
        return any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in pl_module.modules())

    def on_before_accelerator_backend_setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # copy the model before moving it to accelerator device.
        with pl_module._prevent_trainer_and_dataloaders_deepcopy():
            self._average_model = deepcopy(pl_module)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        optimizers = trainer.optimizers
        lr_schedulers = trainer.lr_schedulers

        if len(optimizers) != 1:
            raise MisconfigurationException(
                "SWA currently works with 1 `optimizer`.")

        if len(lr_schedulers) > 1:
            raise MisconfigurationException(
                "SWA currently not supported for more than 1 `lr_scheduler`.")

        self._model_contains_batch_norm = self.pl_module_contains_batch_norm(
            pl_module)

        if self._model_contains_batch_norm:
            # virtually increase max_epochs to perform batch norm update on latest epoch.
            trainer.fit_loop.max_epochs += 1

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int):
        # trainer.global_step starts with 0 (first training batch).
        # If there are 1000 training steps, the final trainer.global_step == 999 here
        if trainer.global_step == self.swa_start:
            # move average model to request device.
            self._average_model = self._average_model.to(pl_module.device)

            optimizers = trainer.optimizers

            for param_group in optimizers[0].param_groups:
                param_group["lr"] = self._swa_lr

            self._swa_scheduler = SWALR(
                optimizers[0],
                swa_lr=1e-6,
                anneal_steps=self._total_steps - self._swa_step_start,
                anneal_strategy=self._annealing_strategy,
                last_epoch=trainer.max_epochs if self._annealing_strategy == "cos" else -1,
            )
            new_scheduler = {
                'scheduler': self._swa_scheduler,
                'reduce_on_plateau': False,
                'opt_idx': None,
                'name': None,
                'monitor': None,
                'interval': 'step',
                'frequency': 1,
            }

            if trainer.lr_schedulers:
                scheduler_cfg = trainer.lr_schedulers[0]
                rank_zero_info(
                    f"Swapping scheduler {scheduler_cfg['scheduler']} for {self._swa_scheduler}")
                trainer.lr_schedulers[0] = new_scheduler
            else:
                trainer.lr_schedulers.append(new_scheduler)

            self.n_averaged = torch.tensor(
                0, dtype=torch.long, device=pl_module.device)

        if self.swa_start <= trainer.global_step < self.swa_end:
            self.update_parameters(self._average_model,
                                   pl_module, self.n_averaged, self.avg_fn)

        # Note: No > here in case the callback is saved with the model and training continues
        # This part never actually gets executed. Maybe this is for batch norm scenario?
        if trainer.global_step == self.swa_end:
            rank_zero_info('Transferring weights')

            # Transfer weights from average model to pl_module
            self.transfer_weights(self._average_model, pl_module)

            # Reset BatchNorm for update
            self.reset_batch_norm_and_save_state(pl_module)

            # There is no need to perform either backward or optimizer.step as we are
            # performing only one pass over the train data-loader to compute activation statistics
            # Therefore, we will virtually increase `num_training_batches` by 1 and skip backward.
            trainer.num_training_batches += 1
            trainer.fit_loop._skip_backward = True
            self._accumulate_grad_batches = trainer.accumulate_grad_batches

            trainer.accumulate_grad_batches = trainer.num_training_batches

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # If there are 1000 training steps, trainer.global_step == 1000 here
        if self._model_contains_batch_norm and trainer.global_step == self.swa_end + 1:
            # BatchNorm epoch update. Reset state
            trainer.accumulate_grad_batches = self._accumulate_grad_batches
            trainer.num_training_batches -= 1
            trainer.fit_loop.max_epochs -= 1
            self.reset_momenta()
        elif trainer.global_step == self.swa_end:
            rank_zero_info('Last SWA step. Transferring weights')
            # Last SWA epoch. Transfer weights from average model to pl_module
            self.transfer_weights(self._average_model, pl_module)

    @staticmethod
    def transfer_weights(src_pl_module: "pl.LightningModule", dst_pl_module: "pl.LightningModule"):
        for src_param, dst_param in zip(src_pl_module.parameters(), dst_pl_module.parameters()):
            dst_param.detach().copy_(src_param.to(dst_param.device))

    def reset_batch_norm_and_save_state(self, pl_module: "pl.LightningModule"):
        """
        Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L140-L154
        """
        self.momenta = {}
        for module in pl_module.modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            module.running_mean = torch.zeros_like(
                module.running_mean, device=pl_module.device, dtype=module.running_mean.dtype
            )
            module.running_var = torch.ones_like(
                module.running_var, device=pl_module.device, dtype=module.running_var.dtype
            )
            self.momenta[module] = module.momentum
            module.momentum = None
            module.num_batches_tracked *= 0

    def reset_momenta(self):
        """
        Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L164-L165
        """
        for bn_module in self.momenta:
            bn_module.momentum = self.momenta[bn_module]

    @staticmethod
    def update_parameters(
        average_model: "pl.LightningModule", model: "pl.LightningModule", n_averaged: torch.LongTensor, avg_fn: _AVG_FN
    ):
        """
        Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L104-L112
        """
        for p_swa, p_model in zip(average_model.parameters(), model.parameters()):
            device = p_swa.device
            p_swa_ = p_swa.detach()
            p_model_ = p_model.detach().to(device)
            src = p_model_ if n_averaged == 0 else avg_fn(
                p_swa_, p_model_, n_averaged.to(device))
            p_swa_.copy_(src)
        n_averaged += 1

    @staticmethod
    def avg_fn(
        averaged_model_parameter: torch.Tensor, model_parameter: torch.Tensor, num_averaged: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L95-L97
        """
        return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (num_averaged + 1)
