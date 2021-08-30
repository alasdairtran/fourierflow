import torch
from torch import Tensor
from torch import linalg as LA
from torchtyping import TensorType


def get_triangle_area(vertex_coords: Tensor) -> Tensor:
    """Compute the area of a triangle given its three vertices.

    For vertices in R^3, we calculate the area with the half cross product
    formula.

    Parameters
    ----------
    vertex_coords : torch.Tensor, shape (..., 3, n_dims)
        The coordinates of the three vertices of the triangles. We can have
        an arbitrary number of leading batch dimensions.

    Returns
    -------
    area : torch.Tensor, shape (...)
        Area of the triangle given by the three vertices.


    Examples
    --------
    >>> coords = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> area = get_triangle_area(vertex_coords)
    torch.Tensor(0.8660254037844386)

    """
    v1, v2, v3 = torch.unbind(vertex_coords, dim=-2)
    n_dims = v1.shape[-1]
    # v1.shape == v2.shape == v3.shape == [..., n_dims]

    if n_dims == 2:
        width = LA.norm(v2 - v1, dim=-1)
        height = LA.norm(v3 - v1, dim=-1)
        area = 0.5 * width * height

    elif n_dims == 3:
        cross = torch.cross(v2 - v1, v3 - v1)
        area = 0.5 * LA.norm(cross, dim=-1)
        # area.shape == [...]

    else:
        raise ValueError(f'Final dimension must be 2 or 3. Got {n_dims}.')

    return area
