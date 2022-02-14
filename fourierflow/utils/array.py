import jax.numpy as jnp
import numpy as np
import torch
import xarray as xr
from jax_cfd.base.grids import Grid as CFDGrid
from jax_cfd.base.grids import GridArray
from jax_cfd.base.resize import downsample_staggered_velocity
from jax_cfd.data.xarray_utils import normalize
from jax_cfd.spectral.utils import vorticity_to_velocity


def grid_correlation(x, y):
    state_dims = ['x', 'y']
    p = normalize(x, state_dims) * normalize(y, state_dims)
    return p.sum(state_dims)


def downsample_vorticity(vorticity, out_size=32, domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi))):
    B, X, Y, T = vorticity.shape

    is_torch_tensor = False
    if torch.is_tensor(vorticity):
        is_torch_tensor = True
        vorticity = vorticity.cpu().numpy()

    in_grid = Grid(shape=[X, Y], domain=domain)
    out_grid = Grid(shape=[out_size, out_size], domain=domain)
    velocity_solve = vorticity_to_velocity(in_grid)
    vorticity_hat = jnp.fft.rfftn(vorticity, axes=(1, 2))
    all_vorticities = []
    for b in range(B):
        vorticities = []
        for t in range(T):
            out = downsample_vorticity_hat(
                vorticity_hat[b, ..., t], velocity_solve, in_grid, out_grid)
            vorticities.append(out['vorticity'])
        vorticities = jnp.stack(vorticities, axis=-1)
        all_vorticities.append(vorticities)
    all_vorticities = jnp.stack(all_vorticities, axis=0)

    if is_torch_tensor:
        # Use torch.tensor instead of torch.from_numpy to fix the error
        # "The given NumPy array is not writeable, and PyTorch does not
        # support non-writeable tensors".
        all_vorticities = torch.tensor(np.asarray(all_vorticities))

    return all_vorticities


def downsample_vorticity_hat(vorticity_hat, velocity_solve, in_grid, out_grid, out_xarray=False):
    # Convert the vorticity field to the velocity field.
    vxhat, vyhat = velocity_solve(vorticity_hat)
    vx = jnp.fft.irfftn(vxhat, axes=(0, 1))
    vy = jnp.fft.irfftn(vyhat, axes=(0, 1))
    velocity = (GridArray(vx, offset=(1, 0.5), grid=in_grid),
                GridArray(vy, offset=(0.5, 1), grid=in_grid))

    # Downsample the velocity field.
    vx, vy = downsample_staggered_velocity(
        in_grid, out_grid, velocity)

    # Convert back to the vorticity field.
    vorticity = velocity_to_vorticity(vx, vy, out_grid)

    if out_xarray:
        coords = {'x': out_grid.axes()[0], 'y': out_grid.axes()[1]}
        vorticity = xr.DataArray(vorticity, coords=coords, dims=('x', 'y'))
        return {'vx': vx, 'vy': vy, 'vorticity': vorticity}

    return {'vx': vx.data, 'vy': vy.data, 'vorticity': vorticity}


def velocity_to_vorticity(vx, vy, grid):
    x, y = grid.axes()
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dv_dx = (jnp.roll(vy.data, shift=-1, axis=0) - vy.data) / dx
    du_dy = (jnp.roll(vx.data, shift=-1, axis=1) - vx.data) / dy
    vorticity = dv_dx - du_dy
    return vorticity


def calculate_time_until(vorticity_corr, threshold=0.95):
    return (vorticity_corr.mean('sample') >= threshold).idxmin('time').rename('time_until')


class Grid(CFDGrid):
    """Fix error when dask supplies all params when reconstructing the grid."""

    def __init__(self, shape, step=None, domain=None):
        if domain is not None:
            super().__init__(shape, domain=domain)
        else:
            super().__init__(shape, step=step)
