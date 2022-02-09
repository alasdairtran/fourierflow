import dataclasses
from typing import Callable, Optional

import jax.numpy as jnp
from jax_cfd.base import forcings, grids
from jax_cfd.spectral import time_stepping
from jax_cfd.spectral import utils as spectral_utils
from jax_cfd.spectral.equations import _get_grid_variable


@dataclasses.dataclass
class NavierStokes2D(time_stepping.ImplicitExplicitODE):
    """Breaks the Navier-Stokes equation into implicit and explicit parts.

    We fix the offset in lines 62 and 63.

    Implicit parts are the linear terms and explicit parts are the non-linear
    terms.

    Attributes:
      viscosity: strength of the diffusion term
      grid: underlying grid of the process
      smooth: smooth the advection term using the 2/3-rule.
      forcing_fn: forcing function, if None then no forcing is used.
      drag: strength of the drag. Set to zero for no drag.
    """
    viscosity: float
    grid: grids.Grid
    drag: float = 0.
    smooth: bool = True
    forcing_fn: Optional[Callable[[grids.Grid], forcings.ForcingFn]] = None
    _forcing_fn_with_grid = None

    def __post_init__(self):
        self.kx, self.ky = self.grid.rfft_mesh()
        self.laplace = (jnp.pi * 2j)**2 * (self.kx**2 + self.ky**2)
        self.filter_ = spectral_utils.circular_filter_2d(self.grid)
        self.linear_term = self.viscosity * self.laplace - self.drag

        # setup the forcing function with the caller-specified grid.
        if self.forcing_fn is not None:
            self._forcing_fn_with_grid = self.forcing_fn(self.grid)

    def explicit_terms(self, vorticity_hat):
        velocity_solve = spectral_utils.vorticity_to_velocity(self.grid)
        vxhat, vyhat = velocity_solve(vorticity_hat)
        vx, vy = jnp.fft.irfftn(vxhat), jnp.fft.irfftn(vyhat)

        grad_x_hat = 2j * jnp.pi * self.kx * vorticity_hat
        grad_y_hat = 2j * jnp.pi * self.ky * vorticity_hat
        grad_x, grad_y = jnp.fft.irfftn(grad_x_hat), jnp.fft.irfftn(grad_y_hat)

        advection = -(grad_x * vx + grad_y * vy)
        advection_hat = jnp.fft.rfftn(advection)

        if self.smooth is not None:
            advection_hat *= self.filter_

        terms = advection_hat

        if self.forcing_fn is not None:
            fx, fy = self._forcing_fn_with_grid((_get_grid_variable(vx, self.grid, offset=(1, 0.5)),
                                                 _get_grid_variable(vy, self.grid, offset=(0.5, 1))))
            fx_hat, fy_hat = jnp.fft.rfft2(fx.data), jnp.fft.rfft2(fy.data)
            terms += spectral_utils.spectral_curl_2d((self.kx, self.ky),
                                                     (fx_hat, fy_hat))

        return terms

    def implicit_terms(self, vorticity_hat):
        return self.linear_term * vorticity_hat

    def implicit_solve(self, vorticity_hat, time_step):
        return 1 / (1 - time_step * self.linear_term) * vorticity_hat
