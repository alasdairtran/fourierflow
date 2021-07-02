import numpy as np
import scipy.io
import torch
from einops import repeat
from torch.utils.data import DataLoader, Dataset
import os

from fourierflow.common import Datastore


@Datastore.register('navier_stokes')
class NavierStokesDatastore(Datastore):
    name = 'navier_stokes'

    def __init__(self, data_path: str, train_size: int, test_size: int,
                 ssr: int, n_steps: int, n_workers: int, batch_size: int,
                 append_pos: bool = True):
        super().__init__()
        self.n_workers = n_workers
        self.batch_size = batch_size

        data = scipy.io.loadmat(os.path.expandvars(data_path))['u'].astype(np.float32)
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


class NavierStokesDataset(Dataset):
    def __init__(self, a, u):
        self.a = a
        self.u = u

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        return (self.a[idx], self.u[idx])
