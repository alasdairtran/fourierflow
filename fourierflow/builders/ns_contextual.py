import os

import h5py
from torch.utils.data import DataLoader, Dataset

from .base import Builder


class NSContextualBuilder(Builder):
    name = 'ns_contextual'

    def __init__(self, data_path: str, ssr: int, k: int, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        data_path = os.path.expandvars(data_path)
        h5f = h5py.File(data_path)

        self.train_dataset = NavierStokesTrainingDataset(h5f['train'], ssr, k)
        self.valid_dataset = NavierStokesDataset(h5f['valid'], ssr, k)
        self.test_dataset = NavierStokesDataset(h5f['test'], ssr, k)

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(self.train_dataset,
                            shuffle=True,
                            drop_last=False,
                            **self.kwargs)
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(self.valid_dataset,
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
        b = idx // self.T
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
