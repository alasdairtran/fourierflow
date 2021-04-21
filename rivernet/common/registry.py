import torch.nn as nn
from allennlp.common.registrable import Registrable
from pytorch_lightning import LightningDataModule, LightningModule


class Datastore(Registrable, LightningDataModule):
    ...


class Experiment(Registrable, LightningModule):
    ...


class Module(Registrable, nn.Module):
    ...
