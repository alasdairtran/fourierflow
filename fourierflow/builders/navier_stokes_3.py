import os

import h5py
import torch
from einops import rearrange, repeat
from einops.einops import rearrange
from torch.utils.data import DataLoader, Dataset

from fourierflow.registries import Builder


@Builder.register('navier_stokes_3')
class NavierStokes3Builder(Builder):
    name = 'navier_stokes_3'

    def __init__(self, data_path: str, ssr: int, k: int, n_workers: int, batch_size: int):
        super().__init__()
        self.n_workers = n_workers
        self.batch_size = batch_size

        data_path = os.path.expandvars(data_path)
        h5f = h5py.File(data_path)

        self.train_dataset = NavierStokesTrainingDataset(h5f['train'], ssr, k)
        self.valid_dataset = NavierStokesDataset(h5f['valid'], ssr, k)
        self.test_dataset = NavierStokesDataset(h5f['test'], ssr, k)

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
    def __init__(self, data, ssr, k):
        self.u = data['u']
        self.f = data['f']
        self.mu = data['mu']
        self.ssr = ssr
        self.k = k
        self.constant_force = len(self.f.shape) == 3

        self.B = self.u.shape[0]
        self.T = self.u.shape[-1] - k

    def __len__(self):
        return self.B * self.T

    def __getitem__(self, idx):
        b = idx // self.B
        t = idx % self.T
        if self.constant_force:
            f = self.f[b, ::self.ssr, ::self.ssr]
        else:
            f = self.f[b, ::self.ssr, ::self.ssr, t + self.k]
        return {
            'x': self.u[b, ::self.ssr, ::self.ssr, t:t+1],
            'y': self.u[b, ::self.ssr, ::self.ssr, t+self.k:t+self.k+1],
            'mu': self.mu[b],
            'f': f,
        }


class NavierStokesDataset(Dataset):
    def __init__(self, data, ssr, k):
        self.u = data['u']
        self.f = data['f']
        self.mu = data['mu']
        self.ssr = ssr
        self.k = k
        self.constant_force = len(self.f.shape) == 3

        self.B = self.u.shape[0]
        self.T = self.u.shape[-1] - k

    def __len__(self):
        return self.B

    def __getitem__(self, b):
        if self.constant_force:
            f = self.f[b, ::self.ssr, ::self.ssr]
        else:
            f = self.f[b, ::self.ssr, ::self.ssr, ::self.k]

        return {
            'data': self.u[b, ::self.ssr, ::self.ssr, ::self.k],
            'mu': self.mu[b],
            'f': f,
        }
