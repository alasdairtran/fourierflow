import jax_cfd.data.xarray_utils as xru
import xarray
from torch.utils.data import DataLoader, Dataset

from .base import Builder


class NavierStokes4Builder(Builder):
    name = 'navier_stokes_4'

    def __init__(self, train_path: str, valid_path: str, n_workers: int, batch_size: int):
        super().__init__()
        self.n_workers = n_workers
        self.batch_size = batch_size

        train_ds = xarray.open_dataset(train_path).thin(time=10)
        train_ds['vorticity'] = xru.vorticity_2d(train_ds)
        train_w = train_ds['vorticity'].values
        train_w = train_w.transpose(0, 2, 3, 1)
        # train_w.shape == [32, 64, 64, 488]

        eval_ds = xarray.open_dataset(valid_path)
        eval_ds['vorticity'] = xru.vorticity_2d(eval_ds)
        eval_w = eval_ds['vorticity'].values
        eval_w = eval_w.transpose(0, 2, 3, 1)
        # eval_w.shape == [32, 64, 64, 488]

        self.train_dataset = NavierStokesTrainingDataset(train_w)
        self.valid_dataset = NavierStokesDataset(eval_w)
        self.test_dataset = NavierStokesDataset(eval_w)

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
    def __init__(self, data):
        self.data = data

        self.B = self.data.shape[0]
        self.T = self.data.shape[-1] - 1

    def __len__(self):
        return self.B * self.T

    def __getitem__(self, idx):
        b = idx // self.T
        t = idx % self.T
        return {
            'x': self.data[b, :, :, t:t+1],
            'y': self.data[b, :, :, t+1:t+2],
        }


class NavierStokesDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.B = self.data.shape[0]

    def __len__(self):
        return self.B

    def __getitem__(self, b):
        return {
            'data': self.data[b],
        }
