import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .base import Builder


class ElasticityBuilder(Builder):
    name = 'elasticity'

    def __init__(self,
                 sigma_path: str,
                 xy_path: str,
                 rr_path: str,
                 train_size: int,
                 valid_size: int,
                 test_size: int,
                 **kwargs):
        super().__init__()
        self.kwargs = kwargs

        # rr is some mysterious feature vector.
        rr = np.load(rr_path)
        rr = torch.tensor(rr, dtype=torch.float).permute(1, 0)
        # rr.shape == [2000, 42]

        # sigma is probably the stress tensor, the thing we measure on the grid.
        sigma = np.load(sigma_path)
        sigma = torch.tensor(sigma, dtype=torch.float).permute(1, 0)
        sigma = sigma.unsqueeze(-1)
        # sigma.shape == [2000, 972, 1]

        # xy is the (x, y) coordinates on the 2D grid.
        xy = np.load(xy_path)
        xy = torch.tensor(xy, dtype=torch.float).permute(2, 0, 1)
        # xy.shape == [2000, 972, 2]

        self.train_dataset = ElasticityDataset(rr=rr[:train_size],
                                               sigma=sigma[:train_size],
                                               xy=xy[:train_size])

        eval_size = valid_size + test_size
        self.valid_dataset = ElasticityDataset(rr=rr[-eval_size:-test_size],
                                               sigma=sigma[-eval_size:-test_size],
                                               xy=xy[-eval_size:-test_size])

        self.test_dataset = ElasticityDataset(rr=rr[-test_size:],
                                              sigma=sigma[-test_size:],
                                              xy=xy[-test_size:])

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
        return None # TODO: Implement me!


class ElasticityDataset(Dataset):
    def __init__(self, rr, sigma, xy):
        self.rr = rr
        self.sigma = sigma
        self.xy = xy

    def __len__(self):
        return self.rr.shape[0]

    def __getitem__(self, idx):
        return {
            'rr': self.rr[idx],
            'sigma': self.sigma[idx],
            'xy': self.xy[idx],
        }
