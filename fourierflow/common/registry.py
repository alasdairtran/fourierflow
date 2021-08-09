from typing import IO
from typing import Callable, Dict, Optional, Type, TypeVar, Union, cast
import os

import torch
import torch.nn as nn
from allennlp.common import Params, Registrable
from allennlp.training.optimizers import Optimizer
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.cloud_io import load as pl_load
from torch.optim.lr_scheduler import _LRScheduler

T = TypeVar("T", bound="Scheduler")


class Datastore(Registrable, LightningDataModule):
    @property
    def batches_per_epochs(self):
        return len(self.train_dataloader())


class Experiment(Registrable, LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # self.log('n_params', n)

    def configure_optimizers(self):
        parameters = [[n, p] for n, p in self.named_parameters()
                      if p.requires_grad]
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


class Module(Registrable, nn.Module):
    ...


class Scheduler(Registrable, _LRScheduler):
    @classmethod
    def from_params(cls: Type[T],
                    params: Params,
                    constructor_to_call: Callable[..., T] = None,
                    constructor_to_inspect: Union[Callable[..., T], Callable[[
                        T], None]] = None,
                    **extras):
        as_registrable = cast(Type[Registrable], cls)
        default_to_first_choice = as_registrable.default_implementation is not None
        choice = params.pop_choice(
            "type",
            choices=as_registrable.list_available(),
            default_to_first_choice=default_to_first_choice)
        subclass, constructor_name = as_registrable.resolve_class_name(choice)
        if not constructor_name:
            constructor_to_inspect = subclass.__init__
            constructor_to_call = subclass  # type: ignore
        else:
            constructor_to_inspect = cast(
                Callable[..., T], getattr(subclass, constructor_name))
            constructor_to_call = constructor_to_inspect

        return subclass(**params, **extras)


Registrable._registry[Scheduler] = {
    'step_lr': (torch.optim.lr_scheduler.StepLR, None),
}
