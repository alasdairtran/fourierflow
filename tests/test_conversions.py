import jax.numpy as jnp
import jax_cfd.base as cfd
import xarray as xr
from jax_cfd.data.xarray_utils import vorticity_2d
from jax_cfd.spectral import utils as spectral_utils

from fourierflow.utils import correlation


def test_convert_vorticity_to_velocity_and_back():
    path = './data/kolmogorov/re_1000/initial_conditions/test.nc'
    ds = xr.open_dataset(path, engine='h5netcdf')

    grid = cfd.grids.Grid(shape=(2048, 2048),
                          domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

    velocity_solve = spectral_utils.vorticity_to_velocity(grid)

    vorticity_1 = ds.isel(sample=0).vorticity

    vorticity_hat = jnp.fft.rfftn(vorticity_1.values)
    vxhat, vyhat = velocity_solve(vorticity_hat)
    vx, vy = jnp.fft.irfftn(vxhat), jnp.fft.irfftn(vyhat)

    x, y = grid.axes()
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dv_dx = (jnp.roll(vy, shift=-1, axis=0) - vy) / dx
    du_dy = (jnp.roll(vx, shift=-1, axis=1) - vx) / dy
    vorticity_2 = dv_dx - du_dy

    vorticity_2 = xr.DataArray(vorticity_2,
                               coords={
                                   'x': grid.axes()[0],
                                   'y': grid.axes()[1],
                               },
                               dims=('x', 'y'))

    rho = correlation(vorticity_1, vorticity_2)
    assert rho > 0.9999
