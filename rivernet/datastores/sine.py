# Adapted from "Neural ODE Processes" by Alexander Norcliffe, Cristian Bodnar,
# Ben Day, Jacob Moss, Pietro Liò

from math import pi

import numpy as np
import torch
import torchcde
from torch.utils.data import DataLoader, Dataset

from rivernet.common import Datastore


@Datastore.register('sine')
class SineDatastore(Datastore):
    name = 'sine'

    def __init__(self,
                 batch_size: int,
                 n_workers: int,
                 **kwargs):
        super().__init__()
        self.n_workers = n_workers
        self.batch_size = batch_size

        self.train_dataset = SineDataset(
            num_samples=2000, seed=98353, **kwargs)
        self.valid_dataset = SineDataset(num_samples=20, seed=64343, **kwargs)
        self.test_dataset = SineDataset(num_samples=20, seed=94813, **kwargs)
        # train_dataset.shape == [1000, 64, 64, 10]

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


class SineDataset(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------
    amplitude_range : tuple of float
            Defines the range from which the amplitude (i.e. a) of the function
            is sampled.

    shift_range : tuple of float
            Defines the range from which the shift (i.e. b) of the function is
            sampled.

    num_samples : int
            Number of samples of the function contained in dataset.

    n_points : int
            Number of points at which to evaluate f(x) for x in [-pi, pi].
    """

    def __init__(self, amplitude_range=(-1., 1.), shift_range=(-.5, .5),
                 freq_range=(1, 1),
                 num_samples=1000, n_points=200, add_cosine=False,
                 std=0.01,
                 forecast_len=80,
                 backcast_len=120,
                 p=0.,
                 seed=None):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.n_points = n_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1
        self.std = std
        self.forecast_len = forecast_len
        self.backcast_len = backcast_len
        self.p = p

        rs = np.random.RandomState(seed)

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        f_min, f_max = freq_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * rs.rand() + a_min
            # Sample random shift
            b = (b_max - b_min) * rs.rand() + b_min
            f = (f_max - f_min) * rs.rand() + f_min
            t = torch.linspace(-pi, pi, n_points)
            mu = a * torch.sin(f * (t - b))
            if add_cosine:
                a = (a_max - a_min) * rs.rand() + a_min
                b = (b_max - b_min) * rs.rand() + b_min
                f = (f_max - f_min) * rs.rand() + f_min
                mu += a * torch.cos(f * (t - b))
            process = mu + self.std * rs.randn(*mu.shape).astype(np.float32)

            t_x = t[:backcast_len]
            t_y = t[-forecast_len:]

            x = process[:backcast_len]
            y = process[-forecast_len:]

            # Turn data into continuous path.
            # Cache the natural cubic spline coefficients.
            x2 = np.stack([t_x, x], axis=1).astype(np.float32)
            coeffs = torchcde.natural_cubic_coeffs(torch.from_numpy(x2))
            # coeffs.shape == [n_steps - 1, 4 * n_channels]

            self.data.append((t, mu, process, t_x, x, coeffs, t_y, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples
