import os

import numpy as np
import scipy.io
import torch
from einops import repeat
from torch.utils.data import DataLoader, Dataset

from .base import Builder


class NSZongyiBuilder(Builder):
    name = 'ns_zongyi'

    def __init__(self, data_path: str, train_size: int, test_size: int,
                 ssr: int, n_steps: int, append_pos: bool = True, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.data_path = data_path

        data = scipy.io.loadmat(os.path.expandvars(data_path))[
            'u'].astype(np.float32)
        data = torch.from_numpy(data)
        a = data[:, ::ssr, ::ssr, :n_steps]
        u = data[:, ::ssr, ::ssr, n_steps:n_steps*2]
        B, X, Y, T = a.shape

        if append_pos:
            # Note that linspace is inclusive of both ends
            ticks = torch.linspace(0, 1, X)
            grid_x = repeat(ticks, 'x -> b x y 1', b=B, y=Y)
            grid_y = repeat(ticks, 'y -> b x y 1', b=B, x=X)

            # Add positional information to inputs
            a = torch.cat([a, grid_x, grid_y], dim=-1)
            # a.shape == [1200, 64, 64, 12]
            # u.shape == [1200, 64, 64, 10]

        self.train_dataset = NavierStokesDataset(
            a[:train_size], u[:train_size])
        self.test_dataset = NavierStokesDataset(
            a[-test_size:], u[-test_size:])
        # train_dataset.shape == [1000, 64, 64, 10]

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

    def inference_data(self):
        data = scipy.io.loadmat(self.data_path)['u'].astype(np.float32)[:512]
        data = torch.from_numpy(data).cuda()
        return {'data': data}


class NavierStokesDataset(Dataset):
    def __init__(self, a, u):
        self.a = a
        self.u = u
        self.times = np.arange(10, 20)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        return {
            'x': self.a[idx],
            'y': self.u[idx],
            'times': self.times,
        }
