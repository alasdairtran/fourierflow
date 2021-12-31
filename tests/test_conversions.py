import jax.numpy as jnp
import jax_cfd.base as cfd
import xarray as xr
from jax_cfd.data.xarray_utils import vorticity_2d
from jax_cfd.spectral import utils as spectral_utils

from fourierflow.utils import correlation


def test_convert_vorticity_to_velocity_and_back():
    path = './data/kolmogorov/re_1000/baseline/256/trajectories.nc'
    ds = xr.open_dataset(path, engine='h5netcdf')

    grid = cfd.grids.Grid(shape=(256, 256),
                          domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

    velocity_solve = spectral_utils.vorticity_to_velocity(grid)

    vorticity_1 = ds.isel(sample=0, time=0).vorticity

    vorticity_hat = jnp.fft.rfftn(vorticity_1.values)
    vxhat, vyhat = velocity_solve(vorticity_hat)
    vx, vy = jnp.fft.irfftn(vxhat), jnp.fft.irfftn(vyhat)
    ds = xr.Dataset(
        data_vars={
            'u': (('x', 'y'), vx),
            'v': (('x', 'y'), vy),
        },
        coords={
            'x': grid.axes()[0],
            'y': grid.axes()[1],
        },
    )

    vorticity_2 = vorticity_2d(ds)

    rho = correlation(vorticity_1, vorticity_2)
    assert rho > 0.9967
