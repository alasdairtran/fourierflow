# Adapted from "Neural ODE Processes" by Alexander Norcliffe, Cristian Bodnar,
# Ben Day, Jacob Moss, Pietro Liò

import os

import hickle as hkl
import numpy as np
import torch

from .base import RDataset


@RDataset.register('bouncing_ball')
class BouncingBallDataset(RDataset):
    '''
    Download dataset from:
    https://www.dropbox.com/sh/q8l6zh2dpb7fi9b/AAAGAPsLuK-j713xCLl45NdTa/bouncing_ball_data?dl=0&subfolder_nav_tracking=1
    '''

    def __init__(self, data_dir='data', n_frames=20):
        Xtr, Ytr, Xval, Yval, Xtest, Ytest = self.load_bball_data(data_dir)
        self.n_frames = n_frames
        self.data = torch.LongTensor(Ytr)[:6000, :n_frames]
        self.t = torch.LongTensor(Xtr)[:6000, :n_frames].unsqueeze(-1)
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
