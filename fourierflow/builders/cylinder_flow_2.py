import enum
import os

import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset

from fourierflow.registries import Builder


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


@Builder.register('cylinder_flow_2')
class CylinderFlow2Builder(Builder):
    """Load the Cylinder Flow dataset.

    For the cylinder flow, the mesh (cells and mesh_pos) is fixed throughout
    all 600 steps.

    """

    name = 'cylinder_flow'

    def __init__(self, data_path: str, n_workers: int, batch_size: int):
        super().__init__()
        self.n_workers = n_workers
        self.batch_size = batch_size

        data_path = os.path.expandvars(data_path)
        h5f = h5py.File(data_path)

        self.train_dataset = CylinderFlowTrainingDataset(h5f['train'], True)
        self.valid_dataset = CylinderFlowTrainingDataset(h5f['valid'], False)
        self.test_dataset = CylinderFlowTrainingDataset(h5f['test'], False)

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


class CylinderFlowTrainingDataset(Dataset):
    def __init__(self, data, sample):
        self.cells = data['cells']
        self.mesh_pos = data['mesh_pos']
        self.node_type = data['node_type']
        self.velocity = data['velocity']
        self.hilbert = data['hilbert']
        self.target_velocity = data['target_velocity']
        self.pressure = data['pressure']
        self.n_cells = data['n_cells']
        self.n_nodes = data['n_nodes']
        self.B, self.T, _, _ = self.velocity.shape
        self.rs = np.random.RandomState(415123)
        self.sample = sample

    def __len__(self):
        return self.B

    def __getitem__(self, b):
        c = self.n_cells[b]
        n = self.n_nodes[b]
        path = self.hilbert[b, 0, :n]

        velocity = self.velocity[b][:, path]
        targets = self.target_velocity[b][:, path]

        if self.sample:
            idx = np.arange(self.T)
            self.rs.shuffle(idx)
            idx = idx[:64]
            velocity = velocity[idx]
            targets = targets[idx]

        return {
            'cells': self.cells[b, :c],
            'mesh_pos': self.mesh_pos[b][path],
            'node_type': self.node_type[b][path],
            'velocity': velocity,
            'path': path,
            'target_velocity': targets,
        }


class CylinderFlowDataset(Dataset):
    def __init__(self, data):
        self.cells = data['cells']
        self.mesh_pos = data['mesh_pos']
        self.node_type = data['node_type']
        self.velocity = data['velocity']
        self.target_velocity = data['target_velocity']
        self.pressure = data['pressure']
        self.frequencies = data['frequencies']
        self.basis = data['basis']
        self.hilbert = data['hilbert']
        self.n_cells = data['n_cells']
        self.n_nodes = data['n_nodes']
        self.B = self.cells[0]
        self.max_len = 1728
        self.rs = np.random.RandomState(14213)

    def __len__(self):
        return self.B

    def __getitem__(self, b):
        c = self.n_cells[b]
        n = self.n_nodes[b]

        hilbert = self.hilbert[b, :, :n]
        # hilbert.shape == [4, n_cells]

        velocity = self.velocity[b, :, :n]
        # hilbert.shape == [4, n_cells]

        mesh_pos = self.mesh_pos[b, :, :n]

        path = hilbert[self.rs.choice(4)]
        # path.shape == [n_cells]

        start = self.rs.choice(len(path) - self.max_len + 1)
        path = path[start:start+self.max_len]
        # path.shape == [max_len]

        features = np.concatenate([velocity[path], mesh_pos[path]], axis=-1)
        # features.shape == [max_len, 4]

        return {
            'cells': self.cells[b, :c],
            'mesh_pos': self.mesh_pos[b, :n],
            'node_type': self.node_type[b, :n],
            'velocity': self.velocity[b, :, :n],
            'target_velocity': self.target_velocity[b, :, :n],
            'pressure': self.pressure[b, :, :n],
            'frequencies': self.frequencies[b],
            'basis': self.basis[b, :n],
            'mass': self.mass[b, :n, :n],
        }
