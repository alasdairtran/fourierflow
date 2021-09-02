import logging

import h5py
import numpy as np
import torch
import typer
from tqdm import tqdm

from fourierflow.modules.sphara import SpharaBasis, TriMesh

logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def basis(data_path: str = 'data/cylinder_flow/cylinder_flow.h5'):
    torch.manual_seed(81823)
    h5f = h5py.File(data_path, 'a')
    compute_basis_for_split(h5f, 'train')
    compute_basis_for_split(h5f, 'valid')
    compute_basis_for_split(h5f, 'test')


def compute_basis_for_split(h5f, split):
    logger.info(f'Computing basis for {split} split.')
    data = h5f[split]
    n_samples = data['mesh_pos'].shape[0]
    max_nodes = data['n_nodes'][...].max()
    n_modes = 256

    frequencies = h5f.create_dataset(
        f'{split}/frequencies', (n_samples, n_modes), np.float32, fillvalue=np.nan)

    basis = h5f.create_dataset(
        f'{split}/basis', (n_samples, max_nodes, n_modes), np.float32, fillvalue=np.nan)

    for i in tqdm(range(n_samples)):
        n = data['n_nodes'][i]
        vertices = torch.from_numpy(data['mesh_pos'][i, :n]).cuda()

        c = data['n_cells'][i]
        triangles = torch.from_numpy(data['cells'][i, :c]).cuda()

        mesh = TriMesh(triangles, vertices)
        sb = SpharaBasis(mesh, mode='fem', k=n_modes)
        f, b = sb.get_basis()

        frequencies[i] = f.cpu().numpy()
        basis[i, :n] = b.cpu().numpy()


if __name__ == "__main__":
    app()
