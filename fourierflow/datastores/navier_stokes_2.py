import numpy as np
import scipy.io
import torch
from einops import repeat
from torch.utils.data import DataLoader, Dataset

from fourierflow.common import Datastore


@Datastore.register('navier_stokes_2')
class NavierStokes2Datastore(Datastore):
    name = 'navier_stokes_2'

    def __init__(self, data_path: str, train_size: int, test_size: int,
                 ssr: int, n_workers: int, batch_size: int):
        super().__init__()
        self.n_workers = n_workers
        self.batch_size = batch_size

        data = scipy.io.loadmat(data_path)['u'].astype(np.float32)
        # For NavierStokes_V1e-5_N1200_T20.mat
        # data.shape == (1200, 64, 64, 20)

        data = torch.from_numpy(data)
        data = data[:, ::ssr, ::ssr]
        B, X, Y, T = data.shape

        self.train_dataset = NavierStokesDataset(
            data[:train_size])
        self.test_dataset = NavierStokesDataset(
            data[-test_size:])
        # train_dataset.shape == [1000, 64, 64, 20]

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
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]
