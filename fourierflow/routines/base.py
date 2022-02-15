from typing import IO, Callable, Dict, List, Optional, Union

import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cloud_io import load as pl_load


class Routine(LightningModule):
    def __init__(self,
                 optimizer,
                 scheduler,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.scheduler = scheduler

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
                                                                str, torch.device, int, Callable]] = None,
                                   remove_keys: Optional[List[str]] = None):
        if map_location is not None:
            checkpoint = pl_load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = pl_load(checkpoint_path,
                                 map_location=lambda storage, loc: storage)

        # Allow us to run super-resolution evaluations
        state_dict = checkpoint['state_dict']
        remove_keys = remove_keys or []
        strict = len(remove_keys) == 0
        for key in remove_keys:
            del state_dict[key]

        self.load_state_dict(state_dict, strict=strict)
