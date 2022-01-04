import jax.numpy as jnp
import xarray as xr
from jax_cfd.base.grids import GridArray
from jax_cfd.base.resize import downsample_staggered_velocity
from jax_cfd.data.xarray_utils import normalize


def grid_correlation(x, y):
    state_dims = ['x', 'y']
    p = normalize(x, state_dims) * normalize(y, state_dims)
    return p.sum(state_dims)


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
    x, y = out_grid.axes()
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dv_dx = (jnp.roll(vy.data, shift=-1, axis=0) - vy.data) / dx
    du_dy = (jnp.roll(vx.data, shift=-1, axis=1) - vx.data) / dy
    vorticity = dv_dx - du_dy

    if out_xarray:
        coords = {'x': out_grid.axes()[0], 'y': out_grid.axes()[1]}
        vorticity = xr.DataArray(vorticity, coords=coords, dims=('x', 'y'))
        return {'vx': vx, 'vy': vy, 'vorticity': vorticity}

    return {'vx': vx.data, 'vy': vy.data, 'vorticity': vorticity}


def calculate_time_until(vorticity_corr, threshold=0.95):
    return (vorticity_corr.mean('sample') >= threshold).idxmin('time').rename('time_until')
