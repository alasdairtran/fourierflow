import functools
import json
import logging
import os

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
def cylinder_flow(in_dir: str = 'data/cylinder_flow',
                  out_path: str = 'data/cylinder_flow/cylinder_flow.h5'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    h5f = h5py.File(out_path, 'a')

    with open(os.path.join(in_dir, 'meta.json'), 'r') as fp:
        meta = json.loads(fp.read())

    process_cylinder_split('train', meta, h5f, in_dir)
    process_cylinder_split('valid', meta, h5f, in_dir)
    process_cylinder_split('test', meta, h5f, in_dir)


def process_cylinder_split(split, meta, h5f, in_dir):
    ds = tf.data.TFRecordDataset(os.path.join(in_dir, split+'.tfrecord'))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)

    params = {
        'noise': 0.02,
        'gamma': 1.0,
        'field': 'velocity',
        'history': False,
        'size': 2,
        'batch': 2,
    }

    ds = add_targets(ds, [params['field']], add_history=params['history'])

    n_cells = []
    n_nodes = []
    n_steps = None
    logger.info(f'Getting size info from {split}.')
    for s in tqdm(ds):
        n_cells.append(s['cells'].shape[1])
        n_nodes.append(s['mesh_pos'].shape[1])
        if not n_steps:
            n_steps = s['cells'].shape[0]

    n_samples = len(n_cells)
    max_cells = max(n_cells)
    max_nodes = max(n_nodes)
    logger.info('Max cells', max_cells)
    logger.info('Max nodes', max_nodes)

    h5f.create_dataset(f'{split}/n_cells', (n_samples), np.int32, n_cells)
    h5f.create_dataset(f'{split}/n_nodes', (n_samples), np.int32, n_nodes)

    shape_1 = (n_samples, n_steps, max_nodes)
    shape_2 = (n_samples, n_steps, max_nodes, 2)
    shape_3 = (n_samples, n_steps, max_cells, 3)

    cells = h5f.create_dataset(
        f'{split}/cells', shape_3, np.int32, fillvalue=-1)

    mesh_pos = h5f.create_dataset(
        f'{split}/mesh_pos', shape_2, np.float32, fillvalue=np.nan)

    node_type = h5f.create_dataset(
        f'{split}/node_type', shape_1, np.int32, fillvalue=-1)

    velocity = h5f.create_dataset(
        f'{split}/velocity', shape_2, np.float32, fillvalue=np.nan)

    target_velocity = h5f.create_dataset(
        f'{split}/target_velocity', shape_2, np.float32, fillvalue=np.nan)

    pressure = h5f.create_dataset(
        f'{split}/pressure', shape_1, np.float32, fillvalue=np.nan)

    logger(f'Writing {split} to disk.')
    for i, sample in tqdm(enumerate(ds)):
        c = n_cells[i]
        n = n_nodes[i]
        cells[i, :, :c] = sample['cells'].numpy()
        mesh_pos[i, :, :n] = sample['mesh_pos'].numpy()
        node_type[i, :, :n] = sample['node_type'].numpy()[..., 0]
        velocity[i, :, :n] = sample['velocity'].numpy()
        target_velocity[i, :, :n] = sample['target|velocity'].numpy()
        pressure[i, :, :n] = sample['pressure'].numpy()[..., 0]


if __name__ == "__main__":
    app()
