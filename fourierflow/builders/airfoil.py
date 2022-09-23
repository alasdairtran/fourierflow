import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .base import Builder


class AirfoilBuilder(Builder):
    name = 'airfoil'

    def __init__(self,
                 x1_path: str,
                 x2_path: str,
                 sigma_path: str,
                 train_size: int,
                 valid_size: int,
                 test_size: int,
                 **kwargs):
        super().__init__()
        self.kwargs = kwargs

        x1 = np.load(x1_path)
        x1 = torch.tensor(x1, dtype=torch.float)
        # x.shape == [2490, 221, 51]

        x2 = np.load(x2_path)
        x2 = torch.tensor(x2, dtype=torch.float)
        # y.shape == [2490, 221, 51]

        x = torch.stack([x1, x2], dim=-1)
        # x.shape == [2490, 221, 51, 2]

        y = np.load(sigma_path)[:, 4]
        y = torch.tensor(y, dtype=torch.float)
        # y.shape == [2490, 221, 51]

        # The following split ensures that the test set is the same as
        # the one used in the original Geo-FNO paper.
        i = train_size
        j = train_size + test_size
        k = train_size + test_size + valid_size

        self.train_dataset = AirfoilDataset(x=x[:i], y=y[:i])
        self.test_dataset = AirfoilDataset(x=x[i:j], y=y[i:j])
        self.valid_dataset = AirfoilDataset(x=x[j:k], y=y[j:k])

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


class AirfoilDataset(Dataset):
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
