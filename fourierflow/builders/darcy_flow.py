import os

import h5py
import numpy as np
import scipy.io
import torch
from einops import repeat
from torch.utils.data import DataLoader, Dataset

from fourierflow.registries import Builder


@Builder.register('darcy_flow')
class DarcyFlowBuilder(Builder):
    name = 'darcy_flow'

    def __init__(self, train_path: str, test_path: str, train_size: int,
                 test_size: int, r: int, n_workers: int, batch_size: int):
        super().__init__()
        h = int(((421 - 1)/r) + 1)
        s = h
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.eps = 0.00001

        train_data = scipy.io.loadmat(os.path.expandvars(train_path))
        test_data = scipy.io.loadmat(os.path.expandvars(test_path))

        x_train = self._read_field(train_data, 'coeff')
        y_train = self._read_field(train_data, 'sol')
        x_test = self._read_field(test_data, 'coeff')
        y_test = self._read_field(test_data, 'sol')

        x_train = x_train[:train_size, ::r, ::r][:, :s, :s]
        y_train = y_train[:train_size, ::r, ::r][:, :s, :s]
        x_test = x_test[:test_size, ::r, ::r][:, :s, :s]
        y_test = y_test[:test_size, ::r, ::r][:, :s, :s]

        # Normalize
        x_mean = x_train.mean(dim=0, keepdims=True)
        x_std = x_train.std(dim=0, keepdims=True) + self.eps
        x_train = (x_train - x_mean) / x_std
        x_test = (x_test - x_mean) / x_std

        y_mean = y_train.mean(dim=0, keepdims=True)
        y_std = y_train.std(dim=0, keepdims=True)

        self.train_dataset = DarcyFlowDataset(x_train, y_train, y_mean, y_std)
        self.test_dataset = DarcyFlowDataset(x_test, y_test, y_mean, y_std)

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


class DarcyFlowDataset(Dataset):
    def __init__(self, x, y, mean, std):
        self.x = x
        self.y = y
        self.mean = mean[0]
        self.std = std[0]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx], self.mean, self.std)
