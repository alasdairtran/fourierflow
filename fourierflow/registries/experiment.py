from typing import IO, Callable, Dict, Optional, Union

import torch
from allennlp.common import Registrable
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cloud_io import load as pl_load


class Experiment(Registrable, LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_start(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.experiment.summary["n_params"] = n

    def configure_optimizers(self):
        parameters = [[n, p] for n, p in self.named_parameters()
                      if p.requires_grad]
        if hasattr(self, 'lr') and self.lr:
            opt = self.optimizer.construct(model_parameters=parameters,
                                           lr=self.lr)
        else:
            opt = self.optimizer.construct(model_parameters=parameters)

        if self.scheduler:
            scheduler = {'scheduler': self.scheduler.construct(optimizer=opt),
                         'name': self.scheduler_config['name'],
                         'interval': self.scheduler_config['interval'],
                         'monitor': self.scheduler_config['monitor'],
                         'frequency': self.scheduler_config['frequency']}

            return [opt], [scheduler]
        else:
            return opt

    def load_lightning_model_state(self,
                                   checkpoint_path: Union[str, IO],
                                   map_location: Optional[Union[Dict[str, str],
                                                                str, torch.device, int, Callable]] = None,
                                   strict: bool = True):
        if map_location is not None:
            checkpoint = pl_load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = pl_load(checkpoint_path,
                                 map_location=lambda storage, loc: storage)

        self.load_state_dict(checkpoint['state_dict'], strict=strict)
