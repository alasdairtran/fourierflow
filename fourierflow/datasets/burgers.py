# Adapted from "Neural ODE Processes" by Alexander Norcliffe, Cristian Bodnar,
# Ben Day, Jacob Moss, Pietro Liò

from math import pi

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

from .base import RDataset


@RDataset.register('burgers')
class BurgersEquation(Dataset):
    """Burgers' equation.

    Parameters
    ----------
    path : int
        Path to the Matlab file.

    ssr : int
        Subsampling rate.

    """

    def __init__(self, path, start, end, ssr=8):

        data = scipy.io.loadmat(path)
        # data['a'].shape == [n_total_samples, grid_size] == [2048, 8192]

        X = torch.from_numpy(data['a'][start:end, ::ssr]).float()
        y = torch.from_numpy(data['u'][start:end, ::ssr]).float()

        # Append location information
        S = 8192 // ssr
        grid = np.linspace(0, 2 * np.pi, S).reshape(1, S, 1)
        grid = torch.from_numpy(grid).float()

        N = X.shape[0]
        X = torch.cat([X.reshape(N, S, 1), grid.repeat(N, 1, 1)], dim=2)
        # X.shape == [n_samples, n_steps, 2]

        self.X = X
        self.y = y

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

    def __len__(self):
        return len(self.X)
