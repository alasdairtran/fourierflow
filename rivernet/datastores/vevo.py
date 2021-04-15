import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .base import Datastore


@Datastore.register('vevo')
class VevoDatastore(Datastore):
    name = 'vevo'

    def __init__(self, data_path: str, train_id_path: str, test_id_path: str,
                 batch_size: int, n_workers: int):
        super().__init__()
        self.n_workers = n_workers
        self.batch_size = batch_size

        views = h5py.File(data_path, 'r')['views'][...]

        # Forward-fill missing values
        mask = views == -1
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        views = views[np.arange(idx.shape[0])[:, None], idx]

        # Fill remaining missing values with 0
        views[views == -1] = 0

        assert (views >= 0).all()

        with open(train_id_path, 'rb') as f:
            train_idx = sorted(pickle.load(f))
            train_views = views[train_idx]

        with open(test_id_path, 'rb') as f:
            test_idx = sorted(pickle.load(f))
            test_views = views[test_idx]

        self.train_dataset = VevoDataset(train_views[:, :49])
        self.valid_dataset = VevoDataset(test_views[:, 7:56])
        self.test_dataset = VevoDataset(test_views[:, 14:63])
        # train_dataset.shape == [1000, 64, 64, 10]

        print('train vevo', len(self.train_dataset))

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


class VevoDataset(Dataset):
    def __init__(self, views):
        self.views = torch.from_numpy(views).float()
        self.log_views = torch.log1p(self.views.clamp(min=0))

    def __len__(self):
        return self.views.shape[0]

    def __getitem__(self, idx):
        return (self.views[idx], self.log_views[idx])
