from allennlp.common import Registrable
from pytorch_lightning import LightningDataModule


class Datastore(Registrable, LightningDataModule):
    @property
    def batches_per_epochs(self):
        return len(self.train_dataloader())
