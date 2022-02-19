import numpy as np
from pytorch_lightning import LightningDataModule


class Builder(LightningDataModule):
    @property
    def batches_per_epochs(self):
        return len(self.train_dataloader())


def collate_jax(sample_list):
    sample = sample_list[0]
    if isinstance(sample, tuple):
        batch = tuple(collate_jax([s[i] for s in sample_list])
                      for i in range(len(sample)))
    elif isinstance(sample, dict):
        batch = {k: collate_jax([s[k] for s in sample_list]) for k in sample}
    else:
        batch = np.stack(sample_list, axis=0)

    return batch
