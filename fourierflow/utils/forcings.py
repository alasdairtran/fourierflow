import jax.numpy as jnp
from jax_cfd.base import forcings, grids


def kolmogorov_forcing_fn(grid: grids.Grid, scale, k) -> forcings.ForcingFn:
    """Constant Kolmogorov forcing function.

    Adapted from jax_cfd.spectral.forcings.kolmogorov_forcing_fn.
    """
    offset = (0, 0)
    _, ys = grid.mesh(offset=offset)
    f = scale * jnp.cos(k * ys)
    f = (grids.GridArray(f, offset, grid),)

    def forcing(v):
        del v  # unused
        return f

    return forcing
