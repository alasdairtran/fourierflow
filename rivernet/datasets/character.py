# Adapted from "Neural ODE Processes" by Alexander Norcliffe, Cristian Bodnar,
# Ben Day, Jacob Moss, Pietro Liò

import os
import os.path as osp

import torch
from scipy.io import loadmat
from six.moves import urllib
from torch.nn.utils.rnn import pad_sequence

from .base import RDataset


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


@RDataset.register('character_trajectories')
class CharacterTrajectoriesDataset(RDataset):
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
            xs.append(torch.FloatTensor(ex[0]))
            ys.append(torch.FloatTensor(ex[1]))
            fs.append(torch.FloatTensor(ex[2]))
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
