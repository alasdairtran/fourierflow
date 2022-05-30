import enum
import os

import h5py
from torch.utils.data import DataLoader, Dataset

from .base import Builder


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


class CylinderFlowBuilder(Builder):
    """Load the Cylinder Flow dataset.
    For the cylinder flow, the mesh (cells and mesh_pos) is fixed throughout
    all 600 steps.
    """

    name = 'cylinder_flow'

    def __init__(self, path: str, **kwargs):
        super().__init__()
        self.kwargs = kwargs

        data_path = os.path.expandvars(path)
        h5f = h5py.File(data_path)

        self.train_dataset = CylinderFlowTrainingDataset(h5f['train'])
        self.valid_dataset = CylinderFlowDataset(h5f['valid'])
        self.test_dataset = CylinderFlowDataset(h5f['test'])

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(self.train_dataset,
                            shuffle=True,
                            **self.kwargs)
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(self.valid_dataset,
                            shuffle=False,
                            **self.kwargs)
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(self.test_dataset,
                            shuffle=False,
                            **self.kwargs)
        return loader


class CylinderFlowTrainingDataset(Dataset):
    def __init__(self, data):
        self.cells = data['cells']
        self.mesh_pos = data['mesh_pos']
        self.node_type = data['node_type']
        self.velocity = data['velocity']
        self.target_velocity = data['target_velocity']
        self.pressure = data['pressure']
        self.n_cells = data['n_cells']
        self.n_nodes = data['n_nodes']
        self.B, self.T, _, _ = self.velocity.shape

    def __len__(self):
        return self.B * self.T

    def __getitem__(self, idx):
        b = idx // self.B
        t = idx % self.T
        c = self.n_cells[b]
        n = self.n_nodes[b]
        return {
            'n_cells': c,
            'n_nodes': n,
            'cells': self.cells[b],  # [c, 3] padded with -1
            'mesh_pos': self.mesh_pos[b],  # [n, 2] nan padded
            'node_type': self.node_type[b],  # [n] padded with -1
            'velocity': self.velocity[b, t],  # [n, 2] nan padded
            'target_velocity': self.target_velocity[b, t],  # [n, 2] nan padded
            # 'pressure': self.pressure[b, t],  # [n] nan padded
        }


class CylinderFlowDataset(Dataset):
    def __init__(self, data):
        self.cells = data['cells']
        self.mesh_pos = data['mesh_pos']
        self.node_type = data['node_type']
        self.velocity = data['velocity']
        self.target_velocity = data['target_velocity']
        self.pressure = data['pressure']
        self.n_cells = data['n_cells']
        self.n_nodes = data['n_nodes']
        self.B = self.velocity.shape[0]

    def __len__(self):
        return self.B

    def __getitem__(self, b):
        c = self.n_cells[b]
        n = self.n_nodes[b]
        return {
            'n_cells': c,
            'n_nodes': n,
            'cells': self.cells[b],  # [c, 3]
            'mesh_pos': self.mesh_pos[b],  # [n, 2]
            'node_type': self.node_type[b],  # [n]
            'velocity': self.velocity[b],  # [t, n, 2]
            'target_velocity': self.target_velocity[b],  # [t, n, 2]
            # 'pressure': self.pressure[b],  # [t, n]
        }
