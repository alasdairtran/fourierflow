from pytorch_lightning import LightningDataModule


class Builder(LightningDataModule):
    @property
    def batches_per_epochs(self):
        return len(self.train_dataloader())
