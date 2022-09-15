import logging
import time
from pathlib import Path

import debugpy
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from einops import repeat
from hydra.utils import instantiate
from typer import Typer

logger = logging.getLogger(__name__)

app = Typer()


@app.callback(invoke_without_command=True)
def main(config_path: Path,
         checkpoint_path: Path,
         debug: bool = False):
    """Re-implement F-FNO in JAX."""
    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        debugpy.listen(5678)
        debugpy.wait_for_client()

    logger.info(f'Loading config from {config_path}...')
    config_dir = config_path.parent
    config_name = config_path.stem
    hydra.initialize(config_path=Path('../..') /
                     config_dir, version_base='1.2')
    config = hydra.compose(config_name, overrides=[])
    routine = instantiate(config.routine)
    routine.load_lightning_model_state(str(checkpoint_path), map_location=None)

    logger.info('Converting params...')
    params = get_params(routine)

    logger.info('Loading data...')
    path = config.builder.test_dataset.init_path
    ds = xr.open_dataset(path, engine='h5netcdf')
    x = jnp.array(ds.vorticity.data)[..., None]
    # x.shape == [batch_size, grid_size, grid_size, 1]

    pos_feats = encode_positions(x.shape[1:-1])
    pos_feats = repeat(pos_feats, '... -> b ...', b=x.shape[0])
    x = jnp.concatenate([x, pos_feats], axis=-1)
    # x.shape == [batch_size, grid_size, grid_size, 3]

    mean = params['normalizer']['mean']
    std = params['normalizer']['std']
    x = (x - mean) / std

    def step(x, _):
        y = ffno(params, x)
        x = jnp.concatenate([y, x[..., 1:]], axis=-1)
        return x, jnp.squeeze(y, axis=-1)

    n_steps = 100

    def trajectory(x):
        trajs = jax.lax.scan(step, x, None, length=n_steps)
        return trajs

    logger.info('Jitting functions...')
    trajectory = jax.vmap(trajectory, in_axes=0, out_axes=0)
    trajectory = jax.jit(trajectory)

    logger.info('Running inference...')
    start = time.time()
    _, trajs = trajectory(x)
    sim_time = config.routine.step_size * n_steps
    elapsed = time.time() - start
    t = elapsed / sim_time / x.shape[0]
    print(trajs.shape, t)


def encode_positions(dim_sizes, low=-1, high=1):
    # dim_sizes is a list of dimensions in all positional/time dimensions
    # e.g. for a 64 x 64 image over 20 steps, dim_sizes = [64, 64, 20]
    def generate_grid(size):
        return jnp.linspace(low, high, size)
    grid_list = list(map(generate_grid, dim_sizes))
    pos = jnp.stack(jnp.meshgrid(*grid_list, indexing='ij'), axis=-1)

    return pos


def get_params(routine):
    params = {}

    in_proj = routine.conv.in_proj
    params['in_proj'] = {
        'weight': to_jnp(in_proj.weight.t()),
        'bias': to_jnp(in_proj.bias),
    }

    params['fourier_weight'] = [
        to_complex_jnp(routine.conv.fourier_weight[0]),
        to_complex_jnp(routine.conv.fourier_weight[1]),
    ]

    params['layers'] = []
    for layer in routine.conv.spectral_layers:
        ff = layer.backcast_ff.layers
        params['layers'].append({
            'in_weight': to_jnp(ff[0][0].weight.t()),
            'in_bias': to_jnp(ff[0][0].bias),
            'out_weight': to_jnp(ff[1][0].weight.t()),
            'out_bias': to_jnp(ff[1][0].bias),
        })

    ff = routine.conv.out
    params['out'] = {
        'in_weight': to_jnp(ff[0].weight.t()),
        'in_bias': to_jnp(ff[0].bias),
        'out_weight': to_jnp(ff[1].weight.t()),
        'out_bias': to_jnp(ff[1].bias),
    }

    count = to_jnp(routine.normalizer.count)
    sum_1 = to_jnp(routine.normalizer.sum)
    sum_2 = to_jnp(routine.normalizer.sum_squared)
    mean = sum_1 / count
    params['normalizer'] = {
        'mean': mean,
        'std': jnp.sqrt(sum_2 / count - mean**2),
    }

    return params


def to_jnp(x):
    return jnp.array(x.detach().numpy())


def to_complex_jnp(x):
    x = jnp.array(x.detach().numpy())
    x = jax.lax.complex(x[..., 0], x[..., 1])
    return x


def ffno(params, x):
    n_layers = len(params['layers'])
    w = params['in_proj']
    x = jnp.dot(x, w['weight']) + w['bias']

    for i in range(n_layers):
        layer_params = {
            'fourier_weight': params['fourier_weight'],
            'feedforward': params['layers'][i],
        }
        out = fourier_layer(layer_params, x)
        x = x + out

    x = feedforward(params['out'], x)

    return x


def fourier_layer(params, inputs):
    M, N, I = inputs.shape
    # inputs.shape == [grid_size, grid_size, hidden_size]

    w_x, w_y = params['fourier_weight']
    n_modes = w_x.shape[2]

    # # Dimension 1 # #
    x_hat = jnp.fft.rfft(inputs, axis=1, norm='ortho')
    # x_hat.shape == [grid_size, grid_size // 2 + 1, hidden_size]

    x_hat = x_hat[:, :n_modes, :]
    # x_hat.shape == [grid_size, n_modes, hidden_size]

    x_out = jnp.zeros((M, N // 2 + 1, I), dtype=jnp.complex64)
    x_partial = jnp.einsum('xyi,ioy->xyo', x_hat, w_x)
    x_out = x_out.at[:, :n_modes].set(x_partial)

    x_feats = jnp.fft.irfft(x_out, axis=1, norm='ortho')
    # x_feats.shape == [grid_size, grid_size, hidden_size]

    # # Dimension 2 # #
    y_hat = jnp.fft.rfft(inputs, axis=0, norm='ortho')
    # y_hat.shape == [grid_size // 2 + 1, grid_size, hidden_size]

    y_hat = y_hat[:n_modes, :, :]
    # x_hat.shape == [grid_size, n_modes, hidden_size]

    n_modes = w_y.shape[2]
    y_out = jnp.zeros((M // 2 + 1, N, I), dtype=jnp.complex64)
    y_partial = jnp.einsum('xyi,iox->xyo', y_hat, w_y)
    y_out = y_out.at[:n_modes].set(y_partial)

    y_feats = jnp.fft.irfft(y_out, axis=0, norm='ortho')
    # y_feats.shape == [grid_size, grid_size, hidden_size]

    # # Combining Dimensions # #
    outputs = x_feats + y_feats
    # outputs.shape == [grid_size, grid_size, hidden_size]

    outputs = feedforward(params['feedforward'], outputs)

    return outputs


def feedforward(params, inputs):
    state = jnp.dot(inputs, params['in_weight']) + params['in_bias']
    state = jax.nn.relu(state)
    state = jnp.dot(state, params['out_weight']) + params['out_bias']
    return state


if __name__ == "__main__":
    app()
