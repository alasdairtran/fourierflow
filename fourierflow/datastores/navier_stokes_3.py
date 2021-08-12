import os

import h5py
import torch
from einops import rearrange, repeat
from einops.einops import rearrange
from torch.utils.data import DataLoader, Dataset

from fourierflow.registry import Datastore


@Datastore.register('navier_stokes_3')
class NavierStokes3Datastore(Datastore):
    name = 'navier_stokes_3'

    def __init__(self, data_path: str, train_size: int, valid_size: int,
                 test_size: int, ssr: int, n_workers: int, batch_size: int):
        super().__init__()
        self.n_workers = n_workers
        self.batch_size = batch_size

        data_path = os.path.expandvars(data_path)
        data = h5py.File(data_path)['u'][...]
        forces = h5py.File(data_path)['f'][...]

        data = torch.from_numpy(data)
        data = data[:, ::ssr, ::ssr]

        forces = torch.from_numpy(forces)
        forces = forces[:, ::ssr, ::ssr]

        self.train_dataset = NavierStokesTrainingDataset(
            data[:train_size], forces[:train_size])

        self.valid_dataset = NavierStokesDataset(
            data[train_size:train_size + valid_size],
            forces[train_size:train_size + valid_size])

        self.test_dataset = NavierStokesDataset(
            data[-test_size:], forces[-test_size:])

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.n_workers,
                            drop_last=False,
                            pin_memory=True)
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(self.valid_dataset,
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


class NavierStokesTrainingDataset(Dataset):
    def __init__(self, data, forces):
        # data.shape == [B, X, Y, T]
        x = data[..., :-1]
        y = data[..., 1:]
        T = x.shape[-1]

        x = rearrange(x, 'b m n t -> (b t) m n 1')
        y = rearrange(y, 'b m n t -> (b t) m n 1')
        forces = repeat(forces, 'b m n -> (b t) m n 1', t=T)
        # x.shape == [19000, 64, 64, 1]

        self.x = x
        self.y = y
        self.forces = forces

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {
            'x': self.x[idx],
            'y': self.y[idx],
            'f': self.forces[idx],
        }


class NavierStokesDataset(Dataset):
    def __init__(self, data, forces):
        self.data = data
        self.forces = forces

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'f': self.forces[idx],
        }
