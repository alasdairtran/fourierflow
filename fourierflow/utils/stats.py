from jax_cfd.data.xarray_utils import normalize


def correlation(x, y):
    state_dims = ['x', 'y']
    p = normalize(x, state_dims) * normalize(y, state_dims)
    return p.sum(state_dims)
