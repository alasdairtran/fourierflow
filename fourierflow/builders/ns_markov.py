import os

import numpy as np
import scipy.io
import torch
from einops import rearrange
from einops.einops import rearrange
from torch.utils.data import DataLoader, Dataset

from .base import Builder


class NSMarkovBuilder(Builder):
    name = 'ns_markov'

    def __init__(self, data_path: str, train_size: int, test_size: int,
                 ssr: int, **kwargs):
        super().__init__()
        self.kwargs = kwargs

        data = scipy.io.loadmat(os.path.expandvars(data_path))[
            'u'].astype(np.float32)
        # For NavierStokes_V1e-5_N1200_T20.mat
        # data.shape == (1200, 64, 64, 20)

        data = torch.from_numpy(data)
        data = data[:, ::ssr, ::ssr]
        B, X, Y, T = data.shape

        self.train_dataset = NavierStokesTrainingDataset(
            data[:train_size])
        self.test_dataset = NavierStokesDataset(
            data[-test_size:])
        # train_dataset.shape == [1000, 64, 64, 20]

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(self.train_dataset,
                            shuffle=True,
                            drop_last=False,
                            **self.kwargs)
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(self.test_dataset,
                            shuffle=False,
                            drop_last=False,
                            **self.kwargs)
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(self.test_dataset,
                            shuffle=False,
                            drop_last=False,
                            **self.kwargs)
        return loader


class NavierStokesTrainingDataset(Dataset):
    def __init__(self, data):
        # data.shape == [B, X, Y, T]
        x = data[..., 1:-1]
        y = data[..., 2:]

        dx = data[..., 1:-1] - data[..., :-2]
        dy = data[..., 2:] - data[..., 1:-1]

        x = rearrange(x, 'b m n t -> (b t) m n 1')
        y = rearrange(y, 'b m n t -> (b t) m n 1')

        dx = rearrange(dx, 'b m n t -> (b t) m n 1')
        dy = rearrange(dy, 'b m n t -> (b t) m n 1')

        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {
            'x': self.x[idx],
            'y': self.y[idx],
            'dx': self.dx[idx],
            'dy': self.dy[idx],
        }


class NavierStokesDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.times = np.arange(0, 20, 1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'times': self.times,
        }
