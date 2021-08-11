import os

import h5py
import numpy as np
import scipy.io
import torch
from einops import repeat
from torch.utils.data import DataLoader, Dataset

from fourierflow.registry import Datastore


@Datastore.register('burgers')
class BurgersDatastore(Datastore):
    name = 'burgers'

    def __init__(self, data_path: str, train_size: int,
                 test_size: int, sub: int, n_workers: int, batch_size: int):
        super().__init__()
        h = 2**13 // sub  # total grid size divided by the subsampling rate
        s = h
        self.n_workers = n_workers
        self.batch_size = batch_size

        data = scipy.io.loadmat(os.path.expandvars(data_path))

        x = self._read_field(data, 'a')[:, ::sub]
        y = self._read_field(data, 'u')[:, ::sub]

        x_train = x[:train_size]
        y_train = y[:train_size]
        x_test = x[-test_size:]
        y_test = y[-test_size:]

        self.train_dataset = BurgersDataset(x_train, y_train)
        self.test_dataset = BurgersDataset(x_test, y_test)

    def _read_field(self, data, field):
        x = data[field]
        x = x.astype(np.float32)
        return torch.from_numpy(x)

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.n_workers,
                            drop_last=False,
                            pin_memory=True)
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(self.test_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.n_workers,
                            drop_last=False,
                            pin_memory=True)
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(self.test_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.n_workers,
                            drop_last=False,
                            pin_memory=True)
        return loader


class BurgersDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
