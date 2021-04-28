# Adapted from "Neural ODE Processes" by Alexander Norcliffe, Cristian Bodnar,
# Ben Day, Jacob Moss, Pietro Liò

import numpy as np
import torch
from scipy import ndimage
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import datasets, transforms


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
        self.t = (torch.arange(time_points)/torch.FloatTensor(10.)).view(-1, 1)

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

        self.data = torch.FloatTensor(dataset)
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
        self.data = torch.FloatTensor(dataset, dtype=torch.float32)
        self.t = (torch.arange(
            dataset.shape[1], dtype=torch.float32).view(-1, 1)/10).repeat([dataset.shape[0], 1, 1])
        self.data = list(zip(self.t, self.data))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
