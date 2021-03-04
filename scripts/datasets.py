# Adapted from "Neural ODE Processes" by Alexander Norcliffe, Cristian Bodnar, Ben Day, Jacob Moss, Pietro Liò

import glob
import os
import os.path as osp
import random
from math import pi

import hickle as hkl
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from scipy.integrate import odeint
from scipy.io import loadmat
from six.moves import urllib
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


class LinearData(Dataset):
    """
    Dataset of functions f(x) = ax + b where a and b are randomly
    sampled. The function is evaluated from 0 to 5.

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

    num_points : int
            Number of points at which to evaluate f(x) for x in [0, 5].
    """

    def __init__(self, grad_range=(-1., 1.), shift_range=(-.5, .5),
                 num_samples=1000, num_points=100):
        self.grad_range = grad_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = grad_range
        b_min, b_max = shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            # Sample random shift
            b = (b_max - b_min) * np.random.rand() + b_min
            # Shape (num_points, x_dim)
            x = torch.linspace(0, 5, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)
            y = a*x + b
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


class SineData(Dataset):
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

    num_points : int
            Number of points at which to evaluate f(x) for x in [-pi, pi].
    """

    def __init__(self, amplitude_range=(-1., 1.), shift_range=(-.5, .5),
                 freq_range=(1, 1),
                 num_samples=1000, num_points=100, add_cosine=False):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        rs = np.random.RandomState(5124)

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
            # Shape (num_points, x_dim)
            x = torch.linspace(-pi, pi, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)
            mu = a * torch.sin(f * (x - b))
            if add_cosine:
                a = (a_max - a_min) * rs.rand() + a_min
                b = (b_max - b_min) * rs.rand() + b_min
                f = (f_max - f_min) * rs.rand() + f_min
                mu += a * torch.cos(f * (x - b))
            y = mu  # + 0.1 * rs.randn(*mu.shape).astype(np.float32)
            self.data.append((x, y, mu))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


class ExpData(Dataset):
    """
    Dataset of functions f(x) = a * exp(x - b) where a and b are randomly
    sampled. The function is evaluated from -1 to 4.

    Parameters
    ----------
    amplitude_range : tuple of float
            Defines the range from which the amplitude (i.e. a) of the exp function
            is sampled.

    shift_range : tuple of float
            Defines the range from which the shift (i.e. b) of the exp function is
            sampled.

    num_samples : int
            Number of samples of the function contained in dataset.

    num_points : int
            Number of points at which to evaluate f(x) for x in [-1, 4].
    """

    def __init__(self, amplitude_range=(-1., 1.), shift_range=(-.5, .5),
                 num_samples=1000, num_points=100):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            # Sample random shift
            b = (b_max - b_min) * np.random.rand() + b_min
            # Shape (num_points, x_dim)
            x = torch.linspace(-1, 4, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)
            y = (a/60) * torch.exp(x - b)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


class DampOscilData(Dataset):
    """
    Dataset of functions f(x) = a * exp(-0.5x)*sin(x-b) where a and b are randomly
    sampled. The function is evaluated from 0 to 5.

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

    num_points : int
            Number of points at which to evaluate f(x) for x in [0, 5].
    """

    def __init__(self, amplitude_range=(-1., 1.), shift_range=(-.5, .5),
                 num_samples=1000, num_points=100):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            # Sample random shift
            b = (b_max - b_min) * np.random.rand() + b_min
            # Shape (num_points, x_dim)
            x = torch.linspace(0, 5, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)
            y = a * torch.sin(x - b) * torch.exp(-0.5*x)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


def download_url(url, folder, log=True):
    """
    Downloads the content of an URL to a specific folder.

    Args:
            url (string): The url.
            folder (string): The folder.
            log (bool, optional): If :obj:`False`, will not print anything to the
                    console. (default: :obj:`True`)
    """

    filename = url.rpartition('/')[2]
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print('Using exist file', filename)
        return path

    if log:
        print('Downloading', url)

    os.mkdir(folder)
    data = urllib.request.urlopen(url)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path


class CharacterTrajectoriesDataset(Dataset):
    """
    CharacterTrajectories dataset.
    """

    def __init__(self, root_dir,
                 position=True, velocity=False,
                 include_length=False, max_length=None):
        """
        args
          root_dir - where to look for the data or download it to
          position - whether to include the position values
          velocity - whether to include the velocity values
          max_length - cutoff for max number of steps, if None, use all (205)
          include_length - whether to include the original sequence length as an output (int)
                                           if a max_length is given, return min(seq_length, max_length)
        """

        url = ('https://archive.ics.uci.edu/ml/machine-learning-databases'
               '/character-trajectories/mixoutALL_shifted.mat')

        self.root_dir = root_dir
        self.include_length = include_length

        self.path_to_data = download_url(url, self.root_dir)

        raw = loadmat(self.path_to_data)
        raw_data = raw['mixout'][0]
        self.int_labels = torch.LongTensor(raw['consts'][0][0][4][0])
        self.label_key = [(i, label[0])
                          for i, label in enumerate(raw['consts'][0][0][3][0])]

        # pad to be the same length
        xs = []
        ys = []
        fs = []
        seq_length = []
        for ex in raw_data:
            xs.append(torch.tensor(ex[0]))
            ys.append(torch.tensor(ex[1]))
            fs.append(torch.tensor(ex[2]))
            seq_length.append(len(ex[0]))

        # need to pad them as separate sequences because it doesn't like padding 3-vectors directly
        padded_xs = pad_sequence(xs).permute(1, 0)
        padded_ys = pad_sequence(ys).permute(1, 0)
        padded_fs = pad_sequence(fs).permute(1, 0)
        self.seq_length = seq_length
        if max_length is None:
            self.max_length = padded_xs.size()[-1]
        else:
            self.max_length = max_length
            padded_xs = padded_xs[:, :self.max_length]
            padded_ys = padded_ys[:, :self.max_length]
            padded_fs = padded_fs[:, :self.max_length]
        # make times
        self.times = torch.linspace(0, 1, self.max_length).unsqueeze(-1)

        # dimensions [batch, timestep, time and 3-velocity in x,y,force]
        padded = torch.stack([padded_xs, padded_ys, padded_fs], dim=2)

        # position is the cumulative sum of velocity over time
        if position and velocity:
            self.states = torch.cat([padded, padded.cumsum(dim=1)], dim=-1)
        elif position:
            self.states = padded.cumsum(dim=1)
        elif velocity:
            self.states = padded
        else:
            raise Exception(
                'Neither position nor velocity selected for Character data.')
        # rescale
        self.states = self.states[:, :, :2]
        self.states = 0.1*self.states.float()

        self.data = []
        ts = torch.linspace(0, 1, 205).float()
        ts = ts.unsqueeze(1)
        for i in range(len(self.states)):
            self.data.append((ts, self.states[i]))
        self.data = self.data[:20000]

    def __getitem__(self, idx):
        if self.include_length:
            return self.times, self.states[idx], min(self.seq_length[idx], self.max_length)
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


class MarkovJumpProcess:
    """
    Implements a generic markov jump process and algorithms for simulating it.
    It is an abstract class, it needs to be inherited by a concrete implementation.
    """

    def __init__(self, init, params):

        self.state = np.asarray(init)
        self.params = np.asarray(params)
        self.time = 0.0

    def _calc_propensities(self):
        raise NotImplementedError(
            'This is an abstract method and should be implemented in a subclass.')

    def _do_reaction(self, reaction):
        raise NotImplementedError(
            'This is an abstract method and should be implemented in a subclass.')

    def sim_steps(self, num_steps):
        """Simulates the process with the gillespie algorithm for a specified number of steps."""

        times = [self.time]
        states = [self.state.copy()]

        for _ in range(num_steps):

            rates = self.params * self._calc_propensities()
            total_rate = rates.sum()

            if total_rate == 0:
                self.time = float('inf')
                break

            self.time += numpy.random.exponential(scale=1/total_rate)

            reaction = self.discrete_sample(rates / total_rate)[0]
            self._do_reaction(reaction)

            times.append(self.time)
            states.append(self.state.copy())

        return times, np.array(states)

    def sim_time(self, dt, duration, max_n_steps=float('inf')):
        """Simulates the process with the gillespie algorithm for a specified time duration."""

        num_rec = int(duration / dt) + 1
        states = np.zeros([num_rec, self.state.size])
        cur_time = self.time
        n_steps = 0

        for i in range(num_rec):

            while cur_time > self.time:

                rates = self.params * self._calc_propensities()
                total_rate = rates.sum()

                if total_rate == 0:
                    self.time = float('inf')
                    break

                exp_scale = max(1/total_rate, 1e-3)
                self.time += np.random.exponential(scale=exp_scale)

                reaction = np.random.multinomial(1, rates / total_rate)
                reaction = np.argmax(reaction)
                self._do_reaction(reaction)

                n_steps += 1
                if n_steps > max_n_steps:
                    raise SimTooLongException(max_n_steps)

            states[i] = self.state.copy()
            cur_time += dt

        return np.array(states)


class LotkaVolterra(MarkovJumpProcess):
    """Implements the lotka-volterra population model."""

    def _calc_propensities(self):

        x, y = self.state
        xy = x * y
        return np.array([xy, x, y, xy])

    def _do_reaction(self, reaction):

        if reaction == 0:
            self.state[0] += 1
        elif reaction == 1:
            self.state[0] -= 1
        elif reaction == 2:
            self.state[1] += 1
        elif reaction == 3:
            self.state[1] -= 1
        else:
            raise ValueError('Unknown reaction.')


class StochasticLotkaVolteraData(Dataset):
    """
    Dataset of time-seires sampled from a Lotka-Voltera model
    ----------
    amplitude_range : tuple of float
            Defines the range from which the amplitude (i.e. a) of the sine function
            is sampled.
    shift_range : tuple of float
            Defines the range from which the shift (i.e. b) of the sine function is
            sampled.
    num_samples : int
            Number of samples of the function contained in dataset.
    num_points : int
            Number of points at which to evaluate f(x) for x in [-pi, pi].
    """

    def __init__(self, initial_X=50, initial_Y=100,
                 num_samples=1000, dt=0.2):
        self.initial_X = initial_X
        self.initial_Y = initial_Y
        self.num_samples = num_samples
        self.x_dim = 1
        self.y_dim = 2
        self.dt = dt

        self.init = [self.initial_X, self.initial_Y]
        self.params = [0.01, 0.5, 1.0, 0.01]
        self.duration = 30

        # Generate data
        self.data = []
        print("Creating dataset...", flush=True)

        removed = 0
        for samples in range(num_samples):
            lv = LotkaVolterra(self.init, self.params)
            states = lv.sim_time(dt, self.duration)
            times = torch.linspace(0.0, self.duration,
                                   int(self.duration / dt) + 1)
            times = times.unsqueeze(1)

            # Ignore outlier populations
            if np.max(states) > 600:
                removed += 1
                continue

            # Scale the population ranges to be closer to the real model
            states = torch.FloatTensor(states) * 1/100
            times = times * 1/20
            self.data.append((times, states))

        self.num_samples -= removed

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


class DeterministicLotkaVolteraData(Dataset):
    """
    Dataset of Lotka-Voltera time series.
      Populations (u,v) evolve according to
            u' = \alpha u - \beta u v
            v' = \delta uv - \gamma v
      with the dataset sampled either with (u_0, v_0) fixed and (\alpha, \beta,
      \gamma, \delta) varied, or varying the initial populations for a fixed set
      of greeks.
    If initial values for (u,v) are provided then the greeks are sampled from
            (0.9,0.05,1.25,0.5) to (1.1,0.15,1.75,1.0)
    if values are provided for the greeks then (u_0 = v_0) is sampled from
            (0.5) to (2.0)
    if both are provided, defaults to initial population mode (greeks vary)
    ----------
    initial_u	: int
            fixed initial value for u
    initial_v	: int
            fixed initial value for v
    fixed_alpha : int
            fixed initial value for \alpha
    fixed_beta	: int
            fixed initial value for \beta
    fixed_gamma : int
            fixed initial value for \gamme
    fixed_delta : int
            fixed initial value for \delta
    end_time : float
            the final time (simulation runs from 0 to end_time)
    steps : int
            how many time steps to take from 0 to end_time
    num_samples : int
            Number of samples of the function contained in dataset.
    """

    def __init__(self, initial_u=None, initial_v=None,
                 alpha=None, beta=None, gamma=None, delta=None,
                 num_samples=1000, steps=150, end_time=15):

        if initial_u is None:
            self.mode = 'greek'
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.delta = delta
        else:
            self.mode = 'population'
            self.initial_u = initial_u
            self.initial_v = initial_v

        print('Lotka-Voltera is in {self.mode} mode.')

        self.num_samples = num_samples
        self.steps = steps
        self.end_time = end_time

        # Generate data
        self.data = []
        print("Creating dataset...", flush=True)

        removed = 0
        for samples in tqdm(range(num_samples)):
            times, states = self.generate_ts()
            # normalise times
            times = torch.FloatTensor(times) / 10
            times = times.unsqueeze(1)

            states = torch.FloatTensor(states)
            if self.mode == 'population':
                states = states / 100
            #states = torch.cat((states, times), dim=-1)

            self.data.append((times, states))

        self.num_samples -= removed

    def generate_ts(self):
        if self.mode == 'population':
            X_0 = np.array([self.initial_u, self.initial_v])
            a = np.random.uniform(0.9, 1.1)
            b = np.random.uniform(0.05, 0.15)
            c = np.random.uniform(1.25, 1.75)
            d = np.random.uniform(0.5, 1.0)
        else:
            equal_pop = np.random.uniform(0.25, 1.)
            X_0 = np.array([2*equal_pop, equal_pop])
            a, b, c, d = self.alpha, self.beta, self.gamma, self.delta

        def dX_dt(X, t=0):
            """ Return the growth rate of fox and rabbit populations. """
            return np.array([a*X[0] - b*X[0]*X[1],
                             -c*X[1] + d*X[0]*X[1]])

        t = np.linspace(0, self.end_time, self.steps)
        X = odeint(dX_dt, X_0, t)

        return t, X

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


class CustomMNISTDataset(Dataset):

    def __init__(self, time_points=15, num_points=None, num_velocity_perms=5):
        '''
        Generates a dataset of rotated MNIST digits at different velocities and initial rotations.

        Parameters
        ----------
        time_points:		number of rotation frames
        num_points:			number of datapoints
        num_velocity_perms: number of velocity permutations per datapoint
        '''
        self.t = (torch.arange(time_points)/torch.tensor(10.)).view(-1, 1)

        data_train = datasets.MNIST("./data/mnist",
                                    download=True,
                                    train=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor()]))

        dataset = list()
        count = 0
        for i, (input, label) in enumerate(data_train):
            if label != 3:
                continue
            if num_points is not None and count >= num_points:
                break
            count += 1

            for velocity_perm in range(num_velocity_perms):
                datapoint = list()
                total_rotation = np.random.uniform(270, 450, 1)[0]
                start_rotation = total_rotation * \
                    np.random.choice(range(time_points)) / time_points

                for t in range(time_points):
                    im_rotate = ndimage.rotate(np.asarray(
                        input[0]), start_rotation+total_rotation*t/time_points, reshape=False)
                    datapoint.append(np.asarray(im_rotate).reshape(-1))

                dataset.append(datapoint)
        dataset = np.stack(dataset)

        self.data = torch.tensor(dataset)
        self.t = self.t.repeat(self.data.shape[0], 1, 1)
        self.data = list(zip(self.t, self.data))

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


class RotNISTDataset(Dataset):
    '''
    Loads the rotated 3s from ODE2VAE paper
    https://www.dropbox.com/s/aw0rgwb3iwdd1zm/rot-mnist-3s.mat?dl=0
    '''

    def __init__(self, data_dir='data'):
        mat = loadmat(data_dir+'/rot-mnist-3s.mat')
        dataset = mat['X'][0]
        dataset = dataset.reshape(dataset.shape[0], dataset.shape[1], -1)
        self.data = torch.tensor(dataset, dtype=torch.float32)
        self.t = (torch.arange(
            dataset.shape[1], dtype=torch.float32).view(-1, 1)/10).repeat([dataset.shape[0], 1, 1])
        self.data = list(zip(self.t, self.data))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class BouncingBallDataset(Dataset):
    '''
    Download dataset from:
    https://www.dropbox.com/sh/q8l6zh2dpb7fi9b/AAAGAPsLuK-j713xCLl45NdTa/bouncing_ball_data?dl=0&subfolder_nav_tracking=1
    '''

    def __init__(self, data_dir='data', n_frames=20):
        Xtr, Ytr, Xval, Yval, Xtest, Ytest = self.load_bball_data(data_dir)
        self.n_frames = n_frames
        self.data = torch.tensor(Ytr)[:6000, :n_frames]
        self.t = torch.tensor(Xtr)[:6000, :n_frames].unsqueeze(-1)
        self.data = list(zip(self.t, self.data))

    def __getitem__(self, index):
        return self.data[index]

    def load_bball_data(self, data_dir, dt=0.1):
        '''This function is taken from https://github.com/cagatayyildiz/ODE2VAE with X and Y flipped'''
        Ytr = hkl.load(os.path.join(data_dir, "training.hkl"))
        Xtr = dt*np.arange(0, Ytr.shape[1], dtype=np.float32)
        Xtr = np.tile(Xtr, [Ytr.shape[0], 1])
        Xval = Yval = Xtest = Ytest = None
        # Yval = hkl.load(os.path.join(data_dir, "val.hkl"))
        # Xval	= dt*np.arange(0, Yval.shape[1], dtype=np.float32)
        # Xval	= np.tile(Xval, [Yval.shape[0], 1])

        # Ytest = hkl.load(os.path.join(data_dir, "test.hkl"))
        # Xtest = dt*np.arange(0, Ytest.shape[1],dtype=np.float32)
        # Xtest = np.tile(Xtest, [Ytest.shape[0],1])

        return Xtr, Ytr, Xval, Yval, Xtest, Ytest

    def __len__(self):
        return len(self.data)


class HafnerData(Dataset):
    '''
    Dataset of GSE100099
    MCF7 cells gamma-irradiated over 24 hours
    p53 is typically the protein of interest
    t=0,1,2,3,4,5,6,7,8,9,10,11,12,24
    '''

    def __init__(self, data_dir):
        target_genes = [
            'KAZN', 'PMAIP1', 'PRKAB1', 'CSNK1G1', 'E2F7', 'SLC30A1',
            'PTP4A1', 'RAP2B', 'SUSD6', 'UBR5-AS1', 'RNF19B', 'AEN', 'ZNF79', 'XPC',
            'FAM212B', 'SESN2', 'DCP1B', 'MDM2', 'GADD45A', 'SESN1', 'CDKN1A', 'BTG2'
        ]
        target_genes.extend([
            'DSCAM', 'C14orf93', 'RPL23AP64', 'RPS6KA5', 'MXD1', 'LINC01560', 'THNSL2',
            'EPAS1', 'ARSD', 'NACC2', 'NEDD9', 'GATS', 'ABHD4', 'BBS1', 'TXNIP',
            'KDM4A', 'ZNF767P', 'LTB4R', 'PI4K2A', 'ZNF337', 'PRKX', 'MLLT11',
            'HSPA4L', 'CROT', 'BAX', 'ORAI3', 'CES2', 'PVT1', 'ZFYVE1', 'PIK3R3',
            'TSPYL2', 'PROM2', 'ZBED5-AS1', 'CCNG1', 'STOM', 'IER5', 'STEAP3',
            'TYMSOS', 'TMEM198B', 'TIGAR', 'ASTN2', 'ANKRA2', 'RRM2B', 'TAP1', 'TP53I3', 'PNRC1',
            'GLS2', 'TMEM229B', 'IKBIP', 'ERCC5', 'KIAA1217', 'DDIT4', 'DDB2', 'TP53INP1'
        ])
        np.random.shuffle(target_genes)
        tfs = ['TP53']

        with open(data_dir+'/t0to24.tsv', 'r', 1) as f:
            contents = f.buffer
            df = pd.read_table(contents, sep='\t', index_col=0)

        columns = ['MCF7, t='+str(t)+' h, IR 10Gy, rep1' for t in range(13)]

        self.genes_df = df[df.index.isin(target_genes)][columns]
        self.genes_df = self.genes_df.reindex(target_genes)
        self.tfs_df = df[df.index.isin(tfs)][columns]

        m = self.genes_df.values
        genes_norm = 1/m.shape[0] * \
            np.linalg.norm(m, axis=1, ord=None)  # l2 norm
        self.genes = torch.tensor(
            m / np.sqrt(genes_norm.reshape(-1, 1)), dtype=torch.float32).unsqueeze(-1)

        f = self.tfs_df.values
        tfs_norm = 1/f.shape[0] * \
            np.linalg.norm(f, axis=1, ord=None)  # l2 norm
        self.tfs = f / np.sqrt(tfs_norm.reshape(-1, 1))

        self.t = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.float32).view(-1, 1)
        self.t = self.t.repeat([self.genes.shape[0], 1, 1])
        self.data = list(zip(self.t, self.genes))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.genes.shape[0]
