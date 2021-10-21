import logging

import h5py
import numpy as np
import torch
import typer
from tqdm import tqdm

from fourierflow.modules.hilbert import linearize
from fourierflow.modules.sphara import SpharaBasis, TriMesh

logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def basis(data_path: str = 'data/cylinder_flow/cylinder_flow.h5',
          mode: str = 'fem',
          n_modes: int = 128):
    torch.manual_seed(81823)
    h5f = h5py.File(data_path, 'a')
    compute_basis_for_split(h5f, 'train', mode, n_modes)
    compute_basis_for_split(h5f, 'valid', mode, n_modes)
    compute_basis_for_split(h5f, 'test', mode, n_modes)


@app.command()
def hilbert(data_path):
    torch.manual_seed(81823)
    h5f = h5py.File(data_path, 'a')
    compute_hilbert_curves_for_split(h5f, 'train')
    compute_hilbert_curves_for_split(h5f, 'valid')
    compute_hilbert_curves_for_split(h5f, 'test')


def compute_basis_for_split(h5f, split, mode, n_modes):
    logger.info(f'Computing basis for {split} split.')
    data = h5f[split]
    n_samples = data['mesh_pos'].shape[0]
    max_nodes = data['n_nodes'][...].max()
    n_modes = n_modes if mode == 'fem' else max_nodes

    if f'{split}/frequencies' not in h5f:
        h5f.create_dataset(f'{split}/frequencies',
                           shape=(n_samples, n_modes),
                           dtype=np.float32,
                           fillvalue=np.nan)

    if f'{split}/basis' not in h5f:
        h5f.create_dataset(f'{split}/basis',
                           shape=(n_samples, max_nodes, n_modes),
                           dtype=np.float32,
                           fillvalue=np.nan)

    if mode == 'fem' and f'{split}/mass' not in h5f:
        h5f.create_dataset(f'{split}/mass',
                           shape=(n_samples, max_nodes, max_nodes),
                           dtype=np.float32,
                           fillvalue=np.nan)

    frequencies = h5f[f'{split}/frequencies']
    basis = h5f[f'{split}/basis']
    if mode == 'fem':
        mass = h5f[f'{split}/mass']

    for i in tqdm(range(n_samples)):
        n = data['n_nodes'][i]
        vertices = torch.from_numpy(data['mesh_pos'][i, :n]).cuda()

        c = data['n_cells'][i]
        triangles = torch.from_numpy(data['cells'][i, :c]).cuda()

        mesh = TriMesh(triangles, vertices)
        sb = SpharaBasis(mesh, mode=mode, k=n_modes)
        f, b = sb.get_basis()

        frequencies[i, :n] = f.cpu().numpy()
        basis[i, :n, :n] = b.cpu().numpy()
        if mode == 'fem':
            mass[i, :n, :n] = sb.mass.to_dense().cpu().numpy()


def compute_hilbert_curves_for_split(h5f, split):
    logger.info(f'Computing Hilbert curves for {split} split.')
    data = h5f[split]
    n_samples = data['mesh_pos'].shape[0]
    max_nodes = data['n_nodes'][...].max()

    if f'{split}/hilbert' not in h5f:
        h5f.create_dataset(f'{split}/hilbert',
                           shape=(n_samples, 4, max_nodes),
                           dtype=np.int32,
                           fillvalue=-1)
    hilbert = h5f[f'{split}/hilbert']

    for i in tqdm(range(n_samples)):
        n = data['n_nodes'][i]
        mesh_pos = data['mesh_pos'][i, :n]

        paths = []
        for shape in 'DUNE':
            indices = list(range(len(mesh_pos)))
            curve = linearize(indices, mesh_pos, shape)
            paths.append(curve.get_path())
        paths = np.array(paths)
        hilbert[i, :, :n] = paths


if __name__ == "__main__":
    app()
