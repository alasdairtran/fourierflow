import pytorch_lightning as pl
from allennlp.common.registrable import Registrable


class Experiment(Registrable, pl.LightningModule):
    pass
