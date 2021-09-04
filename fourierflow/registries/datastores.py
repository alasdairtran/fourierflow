from allennlp.common import Registrable
from pytorch_lightning import LightningDataModule


class Builder(Registrable, LightningDataModule):
    @property
    def batches_per_epochs(self):
        return len(self.train_dataloader())
