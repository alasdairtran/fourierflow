import numpy as np
import scipy.io
import torch
from einops import repeat
from torch.utils.data import DataLoader, Dataset

from .base import Builder


class PlasticityBuilder(Builder):
    name = 'plasticity'

    def __init__(self,
                 data_path: str,
                 train_size: int,
                 valid_size: int,
                 test_size: int,
                 s1: int,
                 s2: int,
                 t: int,
                 **kwargs):
        super().__init__()
        self.kwargs = kwargs

        data = scipy.io.loadmat(data_path)
        x = torch.from_numpy(data['input'].astype(np.float32))
        # x.shape == [987, 101]

        x = repeat(x, 'b s1 -> b s1 s2 t 1', s2=s2, t=t)
        # x.shape == [987, 101, 31, 20, 1]

        y = torch.from_numpy(data['output'].astype(np.float32))
        # y.shape == [987, 101, 31, 20, 4]

        i = train_size
        j = train_size + valid_size
        k = train_size + valid_size + test_size

        self.train_dataset = PlasticityDataset(x=x[:i], y=y[:i])
        self.valid_dataset = PlasticityDataset(x=x[i:j], y=y[i:j])
        self.test_dataset = PlasticityDataset(x=x[j:k], y=y[j:k])

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

    def inference_data(self):
        return None  # TODO: Implement me!


class PlasticityDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {
            'x': self.x[idx],
            'y': self.y[idx],
        }
