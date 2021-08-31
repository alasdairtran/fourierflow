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
    tensor(0.8660254037844386)

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


class TriMesh:
    def __init__(self, triangles: Tensor, vertices: Tensor):
        # Each triangle is a triple of vertex indices.
        self.triangles = triangles
        # triangles.shape == [n_triangles, 3]

        # Each vertex is a coordinates
        self.vertices = vertices
        # vertices.shape == [n_vertices, n_dims]

    def get_mass_matrix(self, mode='normal') -> Tensor:
        """Get the mass matrix of a triangular mesh.

        The method determines a mass matrix of a triangular mesh.

        Parameters
        ----------
        mode : {'normal', 'lumped'}, optional
            The `mode` parameter can be used to select whether a normal mass
            matrix or a lumped mass matrix is to be determined.

        Returns
        -------
        massmatrix : SparseTensor, shape (n_points, n_points)
            Symmetric matrix, which contains the mass values for each edge and
            vertex for the FEM approach. The number of vertices of the
            triangular mesh is n_points.

        Examples
        --------
        >>> triangles = torch.tensor([[0, 1, 2]])
        >>> vertices = torch.tensor([[1.0, 0, 0], [0, 2, 0], [0, 0, 3]])
        >>> mesh = TriMesh(triangles, vertices)
        >>> mesh.get_mass_matrix(mode='normal')
        tensor([[ 0.58333333,  0.29166667,  0.29166667],
                [ 0.29166667,  0.58333333,  0.29166667],
                [ 0.29166667,  0.29166667,  0.58333333]])

        References
        ----------
        :cite:`vallet07,dyer07,zhang07`

        """
        if mode == 'lumped':
            return self._get_lumped_mass_matrix()
        elif mode == 'normal':
            return self._get_normal_mass_matrix()
        else:
            raise ValueError(f'Mode must be lumped or normal. Got {mode}')

    def _get_normal_mass_matrix(self) -> Tensor:
        vertex_coords = torch.embedding(self.vertices, self.triangles)
        # vertex_coords.shape == [n_triangles, 3, n_dims]

        areas = get_triangle_area(vertex_coords)
        a12 = areas / 12
        a6 = areas / 6
        # areas.shape == [n_triangles]

        v1, v2, v3 = torch.unbind(self.triangles, dim=-1)
        # v1.shape == v2.shape == v3.shape == [n_triangles]

        # Each edge of a triangle gets a twelfth of the area.
        # Each vertex (self-loop) of a triangle gets a sixth of the area.
        sources = torch.cat([v1, v1, v2, v2, v3, v3, v1, v2, v3])
        targets = torch.cat([v2, v3, v1, v3, v1, v2, v1, v2, v3])
        indices = torch.stack([sources, targets])
        values = torch.cat([a12, a12, a12, a12, a12, a12, a6, a6, a6])

        n = int(self.triangles.max().item()) + 1
        mass = torch.sparse_coo_tensor(indices, values, size=[n, n],
                                       dtype=areas.dtype, device=areas.device)

        return mass.coalesce()

    def _get_lumped_mass_matrix(self):
        raise NotImplementedError
