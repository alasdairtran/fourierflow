import numpy as np
import torch

from .basis import SpharaBasis


class SpharaFilter(SpharaBasis):
    """SPHARA filter class.

    This class is used to design different types of filters and to
    apply this filter to spatially irregularly sampled data.

    Parameters
    ----------
    mesh : TriMesh object
        A TriMesh object in which the triangulation of the spatial arrangement
        of the sampling points is stored. The SPHARA basic functions are
        determined for this triangulation of the sample points.

    mode : {'unit', 'inv_euclidean', 'fem'}, optional
        The discretization method used to estimate the Laplace-Beltrami
        operator. Using the option 'unit' all edges of
        the mesh are weighted by unit weighting function. The option
        'inv_euclidean' results in edge weights corresponding to the
        inverse Euclidean distance of the edge lengths. The option
        'fem' uses a FEM discretization. The default weighting
        function is 'fem'.

    specification : integer or array, shape (1, n_points)
        If an integer value for specification is passed to the
        constructor, it must be within the interval (-n_points,
        n_points), where n_points is the number of spatial sample
        points. If a positive integer value is passed, a spatial
        low-pass filter with the corresponding number of SPHARA basis
        functions is created, if a negative integer value is passed, a
        spatial low-pass filter is created. If a vector is passed,
        then all SPHARA basis functions corresponding to nonzero
        elements of the vector are used to create the filter. The
        default value of specification is 0, it means a neutral
        all-pass filter is designed and applied.

    """

    def __init__(self, mesh=None, mode='fem', specification=0):
        super().__init__(mesh, mode)
        self.specification = specification
        self._filter = None

    @property
    def specification(self):
        """Get or set the specification of the filter.

        The parameter `specification` has to be an integer or a vector.
        Setting the `specification` will simultaneously apply a plausibility
        check.

        """
        return self._specification

    @specification.setter
    def specification(self, specification):
        if isinstance(specification, int):
            if np.abs(specification) > self.mesh.vertices.shape[0]:
                raise ValueError('Specification is too large.')
            else:
                if specification == 0:
                    self._specification = \
                        torch.ones(self.mesh.vertices.shape[0]).float()
                else:
                    self._specification = \
                        torch.zeros(self.mesh.vertices.shape[0]).float()
                    if specification > 0:
                        self._specification[:specification] = 1
                    else:
                        self._specification[specification:] = 1
        elif isinstance(specification, (list, tuple, np.ndarray)):
            specification = torch.tensor(specification).float()
            if specification.shape[0] != self.mesh.vertices.shape[1]:
                raise IndexError("""The length of the specification vector
                does not match the number of spatial sample points.""")
            else:
                self._specification = specification
        elif isinstance(specification, torch.Tensor):
            self._specification = specification
        else:
            raise TypeError("""The parameter specification has to be
            int or a vector.""")

    def filter(self, data):
        r"""Perform the SPHARA filtering.

        This method performs the spatial SPHARA filtering
        for data defined at spatially distributed sampling points
        described by a triangular mesh. The filtering is
        performed by matrix multiplication of the data matrix and a
        precalculated filter matrix.

        Parameters
        ----------
        data : array, shape(m, n_points)
            A matrix with data to be filtered by spatial SPHARA
            filter. The number of vertices of the triangular mesh is
            n_points. The order of the spatial sample points must
            correspond to that in the vertex list used to determine
            the SPHARA basis functions.

        Returns
        -------
        data_filtered : array, shape (m, n_points)
            A matrix containing the filtered data.

        Examples
        --------
        >>> mesh = TriMesh([[0, 1, 2]], [[1., 0, 0], [0, 2, 0], [0, 0, 3]])
        >>> sf = SpharaFilter(mesh, mode='inv_euclidean', specification=[1., 1., 0.])
        >>> data = torch.randn(5, 3)
        >>> data_filtered = sf.filter(data)

        """
        if self.basis is None or self.frequencies is None:
            self.construct_basis()

        if self._filter is None:
            basis = self.basis @ torch.diag(self.specification)
            if self.mode == 'fem':
                self._filter = self.mass.to_dense() @ basis @ basis.t()
            else:
                self._filter = basis @ basis.t()

        # Does the number of spatial samples of the data correspond to
        # that of the basis functions?
        if self.basis.shape[0] != data.shape[1]:
            raise ValueError('Dimension of data and filter matrix not equal.')

        # filter the data
        data_filtered = data @ self._filter

        return data_filtered
