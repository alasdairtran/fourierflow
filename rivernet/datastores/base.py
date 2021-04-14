import torch
from allennlp.common.registrable import Registrable
from pytorch_lightning import LightningDataModule


class Datastore(Registrable, LightningDataModule):
    pass
