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
    v0, v1, v2 = torch.unbind(vertex_coords, dim=-2)
    n_dims = v1.shape[-1]
    # v1.shape == v2.shape == v3.shape == [..., n_dims]

    if n_dims == 2:
        width = LA.norm(v1 - v0, dim=-1)
        height = LA.norm(v2 - v0, dim=-1)
        area = 0.5 * width * height

    elif n_dims == 3:
        cross = torch.cross(v1 - v0, v2 - v0)
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

        self.n_vertices = int(self.triangles.max().item()) + 1
        self.vertex_coords = torch.embedding(self.vertices, self.triangles)
        # vertex_coords.shape == [n_triangles, 3, n_dims]

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
        areas = get_triangle_area(self.vertex_coords)
        a12 = areas / 12
        a6 = areas / 6
        # areas.shape == [n_triangles]

        n0, n1, n2 = torch.unbind(self.triangles, dim=-1)
        # n0.shape == n1.shape == n2.shape == [n_triangles]

        # Each edge of a triangle gets a twelfth of the area.
        # Each vertex (self-loop) of a triangle gets a sixth of the area.
        sources = torch.cat([n0, n0, n1, n1, n2, n2, n0, n1, n2])
        targets = torch.cat([n1, n2, n0, n2, n0, n1, n0, n1, n2])
        indices = torch.stack([sources, targets])
        values = torch.cat([a12, a12, a12, a12, a12, a12, a6, a6, a6])

        N = self.n_vertices
        mass = torch.sparse_coo_tensor(indices, values, size=[N, N],
                                       dtype=areas.dtype, device=areas.device)

        return mass.coalesce()

    def _get_lumped_mass_matrix(self):
        raise NotImplementedError

    def get_weight_matrix(self, mode='inv_euclidean'):
        """Compute the weight matrix for a triangular mesh.

        The method creates a weighting matrix for the edges of a triangular
        mesh using different weighting function.

        Parameters
        ----------
        mode : {'unit', 'inv_euclidean', 'half_cotangent'}, optional
            The parameter `mode` specifies the method for determining
            the edge weights. Using the option 'unit' all edges of the
            mesh are weighted by unit weighting function, the result
            is an adjacency matrix. The option 'inv_euclidean' results
            in edge weights corresponding to the inverse Euclidean
            distance of the edge lengths. The option 'half_cotangent'
            uses the half of the cotangent of the two angles opposed
            to an edge as weighting function. the default weighting
            function is 'inv_euclidean'.

        Returns
        -------
        weight : tensor, shape (n_points, n_points)
            Symmetric matrix, which contains the weight of the edges
            between adjacent vertices. The number of vertices of the
            triangular mesh is n_points.

        Examples
        --------
        >>> mesh = TriMesh([[0, 1, 2]], [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        >>> mesh.get_weight_matrix(mode='inv_euclidean')
        tensor([[ 0.        ,  0.4472136 ,  0.31622777],
                [ 0.4472136 ,  0.        ,  0.2773501 ],
                [ 0.31622777,  0.2773501 ,  0.        ]])

        """
        if mode == 'unit':
            return self._get_unit_weight_matrix()
        elif mode == 'inv_euclidean':
            return self._get_inv_euclidean_weight_matrix()
        elif mode == 'half_cotangent':
            return self._get_half_cotangent_weight_matrix()
        else:
            raise ValueError(f'Unrecognized mode: {mode}')

    def _get_unit_weight_matrix(self):
        n0, n1, n2 = torch.unbind(self.triangles, dim=-1)
        ones = torch.full(n0.shape, 1.0).to(n0.device)

        sources = torch.cat([n0, n1, n0, n2, n1, n2])
        targets = torch.cat([n1, n0, n2, n0, n2, n1])
        indices = torch.stack([sources, targets])
        values = torch.cat([ones, ones, ones, ones, ones, ones])

        N = self.n_vertices
        weight = torch.sparse_coo_tensor(indices, values, size=[N, N])

        return weight.coalesce()

    def _get_inv_euclidean_weight_matrix(self):
        n0, n1, n2 = torch.unbind(self.triangles, dim=-1)
        v0, v1, v2 = torch.unbind(self.vertex_coords, dim=-2)

        e10 = 1 / LA.norm(v1 - v0, dim=-1)
        e20 = 1 / LA.norm(v2 - v0, dim=-1)
        e21 = 1 / LA.norm(v2 - v1, dim=-1)

        sources = torch.cat([n0, n1, n0, n2, n2, n1])
        targets = torch.cat([n1, n0, n2, n0, n1, n2])
        indices = torch.stack([sources, targets])
        values = torch.cat([e10, e10, e20, e20, e21, e21])

        N = self.n_vertices
        weight = torch.sparse_coo_tensor(indices, values, size=[N, N])

        return weight.coalesce()

    def _get_half_cotangent_weight_matrix(self):
        n0, n1, n2 = torch.unbind(self.triangles, dim=-1)
        v0, v1, v2 = torch.unbind(self.vertex_coords, dim=-2)

        def get_edge_weight(x, y):
            num = torch.einsum('bi,bi->b', x, y)  # dot product
            denom = LA.norm(x, dim=-1) * LA.norm(y, dim=-1)
            return 0.5 * (1 / torch.tan(torch.acos(num / denom)))

        w0 = get_edge_weight(v1 - v0, v2 - v0)
        w1 = get_edge_weight(v0 - v1, v2 - v1)
        w2 = get_edge_weight(v0 - v2, v1 - v2)

        sources = torch.cat([n1, n2, n0, n2, n0, n1])
        targets = torch.cat([n2, n1, n2, n0, n1, n0])
        indices = torch.stack([sources, targets])
        values = torch.cat([w0, w0, w1, w1, w2, w2])

        N = self.n_vertices
        weight = torch.sparse_coo_tensor(indices, values, size=[N, N])

        return weight.coalesce()
