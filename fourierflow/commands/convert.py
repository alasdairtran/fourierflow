import functools
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import typer
from tqdm import tqdm

logger = logging.getLogger(__name__)
app = typer.Typer()


def _parse(proto, meta):
    """Parse a trajectory from tf.Example."""
    feature_lists = {k: tf.io.VarLenFeature(tf.string)
                     for k in meta['field_names']}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta['features'].items():
        data = tf.io.decode_raw(
            features[key].values, getattr(tf, field['dtype']))
        data = tf.reshape(data, field['shape'])
        if field['type'] == 'static':
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
        elif field['type'] == 'dynamic_varlen':
            length = tf.io.decode_raw(features['length_'+key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field['type'] != 'dynamic':
            raise ValueError('invalid data format')
        out[key] = data
    return out


def add_targets(ds, fields, add_history):
    """Add target and optionally history fields to dataframe."""
    def fn(trajectory):
        out = {}
        for key, val in trajectory.items():
            out[key] = val[1:-1]
            if key in fields:
                if add_history:
                    out['prev|'+key] = val[0:-2]
                out['target|'+key] = val[2:]
        return out
    return ds.map(fn, num_parallel_calls=8)


@app.command()
def flag_simple():
    raise NotImplementedError


@app.command()
def cylinder_flow(data_dir: str = 'data/meshgraphnets/cylinder_flow',
                  out: str = 'data/meshgraphnets/cylinder_flow/cylinder_flow.h5'):
    out_path = Path(out)
    out_path.parent.mkdir(exist_ok=True)
    h5f = h5py.File(out_path, 'a')

    in_path = Path(data_dir)
    logger.info(f'Reading metadata from {in_path}')
    with open(in_path / 'meta.json', 'r') as fp:
        meta = json.loads(fp.read())

    process_cylinder_split('train', meta, h5f, in_path)
    process_cylinder_split('valid', meta, h5f, in_path)
    process_cylinder_split('test', meta, h5f, in_path)


def process_cylinder_split(split, meta, h5f, in_path):
    logger.info(f'Reading TFRecord from {in_path}')
    ds = tf.data.TFRecordDataset(in_path / f'{split}.tfrecord')
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)

    params = {
        'noise': 0.02,
        'gamma': 1.0,
        'field': 'velocity',
        'history': False,
        'size': 2,
        'batch': 2,
    }

    ds = add_targets(ds, [params['field']], params['history'])

    n_cells = []
    n_nodes = []
    n_steps = None
    logger.info(f'Getting size info from {split}.')
    for s in tqdm(ds):
        n_cells.append(s['cells'].shape[1])
        n_nodes.append(s['mesh_pos'].shape[1])
        if not n_steps:
            n_steps = s['cells'].shape[0]

    # Each sample has 598 time steps.
    n_samples = len(n_cells)
    max_cells = max(n_cells)
    max_nodes = max(n_nodes)
    logger.info(f'Max cells: {max_cells}')
    logger.info(f'Max nodes: {max_nodes}')

    h5f.create_dataset(f'{split}/n_cells', (n_samples,), np.int32, n_cells)
    h5f.create_dataset(f'{split}/n_nodes', (n_samples,), np.int32, n_nodes)

    shape_1 = (n_samples, max_nodes)
    shape_2 = (n_samples, max_nodes, 2)
    shape_3 = (n_samples, max_cells, 3)

    shape_1t = (n_samples, n_steps, max_nodes)
    shape_2t = (n_samples, n_steps, max_nodes, 2)

    # Each cell is a triangle, a triple containing three node indices.
    cells = h5f.create_dataset(
        f'{split}/cells', shape_3, np.int32, fillvalue=-1)

    # Each position is the (x, y) coordinate of a node (a vertex of triangle).
    mesh_pos = h5f.create_dataset(
        f'{split}/mesh_pos', shape_2, np.float32, fillvalue=np.nan)

    # Each node has a type: 0 (normal), 4 (inflow), 5 (outflow), 6 (wall)
    node_type = h5f.create_dataset(
        f'{split}/node_type', shape_1, np.int32, fillvalue=-1)

    # The (x, y) velocity at each node. Can be negative.
    velocity = h5f.create_dataset(
        f'{split}/velocity', shape_2t, np.float32, fillvalue=np.nan)

    # The (x, y) velocity at each node at the next time step.
    target_velocity = h5f.create_dataset(
        f'{split}/target_velocity', shape_2t, np.float32, fillvalue=np.nan)

    # Pressure is a scalar value at each node at each time step.
    pressure = h5f.create_dataset(
        f'{split}/pressure', shape_1t, np.float32, fillvalue=np.nan)

    logger.info(f'Writing {split} to disk.')
    for i, sample in tqdm(enumerate(ds)):
        c = n_cells[i]
        n = n_nodes[i]

        # For cylinder-flow, mesh_pos, node_type, and cells are constant
        # across all time steps.
        cells[i, :c] = sample['cells'].numpy()[0]
        mesh_pos[i, :n] = sample['mesh_pos'].numpy()[0]
        node_type[i, :n] = sample['node_type'].numpy()[0, ..., 0]

        velocity[i, :, :n] = sample['velocity'].numpy()
        target_velocity[i, :, :n] = sample['target|velocity'].numpy()
        pressure[i, :, :n] = sample['pressure'].numpy()[..., 0]


def verify_constant_mesh(h5f):
    verify_constant_mesh_split(h5f['train'])
    verify_constant_mesh_split(h5f['valid'])
    verify_constant_mesh_split(h5f['test'])


def verify_constant_mesh_split(data):
    # Check that the mesh is the same at each time step
    n_samples = data['mesh_pos'].shape[0]
    for i in tqdm(range(n_samples)):
        n_cells = data['n_cells'][i]
        n_nodes = data['n_nodes'][i]

        cells = data['cells'][i, :, :n_cells]
        mesh_pos = data['mesh_pos'][i, :, :n_nodes]
        node_type = data['node_type'][i, :, :n_nodes]

        for t in range(598):
            assert (cells[0] == cells[t]).all()
            assert (mesh_pos[0] == mesh_pos[t]).all()
            assert (node_type[0] == node_type[t]).all()


if __name__ == "__main__":
    app()
