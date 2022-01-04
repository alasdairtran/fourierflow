import jax.numpy as jnp
import jax_cfd.base as cfd
import xarray as xr
from jax_cfd.spectral import utils as spectral_utils

from fourierflow.utils import correlation, downsample_vorticity_hat


def test_convert_vorticity_to_velocity_and_back():
    path = './data/kolmogorov/re_1000/initial_conditions/test.nc'
    ds = xr.open_dataset(path, engine='h5netcdf')

    grid = cfd.grids.Grid(shape=(2048, 2048),
                          domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

    velocity_solve = spectral_utils.vorticity_to_velocity(grid)

    vorticity_1 = ds.isel(sample=0).vorticity

    vorticity_hat = jnp.fft.rfftn(vorticity_1.values, axes=(0, 1))
    vxhat, vyhat = velocity_solve(vorticity_hat)
    vx = jnp.fft.irfftn(vxhat, axes=(0, 1))
    vy = jnp.fft.irfftn(vyhat, axes=(0, 1))

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


def test_repeated_downsampling():
    path = './data/kolmogorov/re_1000/initial_conditions/test.nc'
    ds = xr.open_dataset(path, engine='h5netcdf')
    vorticity_2048 = ds.isel(sample=0).vorticity
    vorticity_2048_hat = jnp.fft.rfftn(vorticity_2048.values, axes=(0, 1))

    domain = ((0, 2 * jnp.pi), (0, 2 * jnp.pi))
    grid_2048 = cfd.grids.Grid(shape=(2048, 2048), domain=domain)
    grid_1024 = cfd.grids.Grid(shape=(1024, 1024), domain=domain)
    grid_512 = cfd.grids.Grid(shape=(512, 512), domain=domain)
    grid_256 = cfd.grids.Grid(shape=(256, 256), domain=domain)
    grid_128 = cfd.grids.Grid(shape=(128, 128), domain=domain)
    grid_64 = cfd.grids.Grid(shape=(64, 64), domain=domain)
    grid_32 = cfd.grids.Grid(shape=(32, 32), domain=domain)
    velocity_solve_2048 = spectral_utils.vorticity_to_velocity(grid_2048)

    # We suffer up to 8% correlation loss when doing repeated downsampling!
    grids = [grid_1024, grid_512, grid_256, grid_128, grid_64, grid_32]
    ref_rhos = [0.9999999, 0.999, 0.998, 0.99, 0.97, .927]

    # Keep halving 2048x2048 grid until we get a 32x32 grid.
    grid_prev = grid_2048
    vorticity_hat = vorticity_2048_hat
    for ref_rho, grid in zip(ref_rhos, grids):
        velocity_solve = spectral_utils.vorticity_to_velocity(grid_prev)
        vorticity = downsample_vorticity_hat(
            vorticity_hat, velocity_solve, grid_prev, grid, True)['vorticity']

        # Directly downsample from 2048x2048 grid.
        vorticity_direct = downsample_vorticity_hat(
            vorticity_2048_hat, velocity_solve_2048, grid_2048, grid, True)['vorticity']

        rho = correlation(vorticity_direct, vorticity)
        assert rho > ref_rho

        # Prepare inputs for next iteration
        vorticity_hat = jnp.fft.rfftn(vorticity.values, axes=(0, 1))
        grid_prev = grid
