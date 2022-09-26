from typing import IO, Callable, Dict, List, Optional, Union

import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cloud_io import load as pl_load


class Routine(LightningModule):
    def __init__(self,
                 optimizer,
                 scheduler,
                 automatic_optimization: bool = True,
                 accumulate_grad_batches: int = 1,
                 clip_val: Optional[float] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.automatic_optimization = automatic_optimization
        self.accumulate_grad_batches = accumulate_grad_batches
        self.clip_val = clip_val

    def warmup(self):
        pass

    def optimize_manually(self, loss, batch_idx):
        if not self.automatic_optimization:
            if self.accumulate_grad_batches == 1:
                opt = self.optimizers()
                opt.zero_grad()
                self.manual_backward(loss)
                if self.clip_val:
                    for group in opt.param_groups:
                        torch.nn.utils.clip_grad_value_(group["params"],
                                                        self.clip_val)
                opt.step()

            else:
                opt = self.optimizers()
                loss /= self.accumulate_grad_batches
                self.manual_backward(loss)
                if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                    if self.clip_val:
                        for group in opt.param_groups:
                            torch.nn.utils.clip_grad_value_(group["params"],
                                                            self.clip_val)
                    opt.step()
                    opt.zero_grad()

            sch = self.lr_schedulers()
            sch.step()

    def infer(self, data):
        with torch.no_grad():
            return self.forward(data)

    def convert_data(self, data):
        data = {k: torch.from_numpy(v).cuda() for k, v in data.items()}
        return data

    def on_train_start(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if self.logger:
            self.logger.experiment.summary["n_params"] = n

    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        if hasattr(self, 'lr') and self.lr:
            opt = self.optimizer(parameters, lr=self.lr)
        else:
            opt = self.optimizer(parameters)

        sch = OmegaConf.to_container(self.scheduler)
        sch['scheduler'] = sch['scheduler'](optimizer=opt)

        return [opt], [sch]

    def load_lightning_model_state(self,
                                   checkpoint_path: Union[str, IO],
                                   map_location: Optional[Union[Dict[str, str],
                                                                str, torch.device, int, Callable]] = None):
        if map_location is not None:
            checkpoint = pl_load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = pl_load(checkpoint_path,
                                 map_location=lambda storage, loc: storage)

        # Allow us to run super-resolution evaluations
        REMOVE_KEYS = ['kx', 'ky', 'lap',
                       'kx_32', 'ky_32', 'lap_32',
                       'kx_64', 'ky_64', 'lap_64',
                       'kx_128', 'ky_128', 'lap_128',
                       'kx_256', 'ky_256', 'lap_256']
        state_dict = checkpoint['state_dict']
        strict = True
        for key in REMOVE_KEYS:
            if key in state_dict:
                del state_dict[key]
                strict = False

        self.load_state_dict(state_dict, strict=strict)
