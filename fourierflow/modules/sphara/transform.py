import numpy as np

from .basis import SpharaBasis


class SpharaTransform(SpharaBasis):
    """SPHARA transform class.

    This class is used to perform the SPHARA forward (analysis) and
    inverse (synthesis) transformation.

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
        'fem' uses a FEM discretization. the default weighting
        function is 'fem'.

    """

    def __init__(self, mesh=None, mode='fem'):
        super().__init__(mesh, mode)

    def analyze(self, data):
        r"""Perform the SPHARA transform (analysis).

        This method performs the SPHARA transform (analysis) of data
        defined at spatially distributed sampling points described by
        a triangular mesh. The forward transformation is performed by
        matrix multiplication of the data matrix and the matrix with
        SPHARA basis functions :math:`\tilde{X} = X \cdot S`, with the
        SPHARA basis :math:`S`, the data matrix :math:`X` and the
        SPHARA coefficients matrix :math:`\tilde{X}`. In the forward
        transformation using SPHARA basic functions determined by
        discretization with FEM approach, the modified scalar product
        including the mass matrix is used :math:`\tilde{X} = X \cdot B
        \cdot S`, with the mass matrix :math:`B`.

        Parameters
        ----------
        data : tensor, shape (batch_size, n_points)
            A matrix with data to be transformed (analyzed) by
            SPHARA. The number of vertices of the triangular mesh is
            n_points. The order of the spatial sample points must
            correspond to that in the vertex list used to determine
            the SPHARA basis functions.

        Returns
        -------
        coefficients : tensor, shape (batch_size, n_points)
            A matrix containing the SPHARA coefficients. The coefficients
            are sorted column by column with increasing spatial frequency,
            starting with DC in the first column.

        Examples
        --------
        >>> mesh = TriMesh([[0, 1, 2]], [[1., 0, 0], [0, 2, 0], [0, 0, 3]])
        >>> st = SpharaTransform(mesh, mode='inv_euclidean')
        >>> basis = st.get_basis()[1]
        >>> samples = torch.tensor([[0., 0, 0], [1, 1, 1]])
        >>> data = torch.cat([samples, basis.t()])
        >>> coefficients = st.analyze(data)
        >>> coefficients
        tensor([[0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                [-1.7320509e+00, -1.7881393e-07,  1.6391277e-07],
                [1.0000001e+00, -3.2789412e-08, -1.1624118e-07],
                [-3.2789412e-08,  1.0000002e+00,  9.2175775e-08],
                [-1.1624118e-07,  9.2175775e-08,  1.0000001e+00]])

        """
        if self.frequencies is None or self.basis is None:
            self.construct_basis()

        # Does the number of spatial samples of the data correspond to
        # that of the basis functions?
        if self.basis.shape[0] != data.shape[1]:
            raise ValueError('Dimension mismatch')

        # Compute the SPHARA coefficients
        if self.mode == 'fem':
            coefficients = data @ self.mass.to_dense() @ self.basis
        else:
            coefficients = data @ self.basis

        return coefficients

    def synthesize(self, coefficients):
        r"""Perform the inverse SPHARA transform (synthesis).

        This method performs the inverse SPHARA transform (synthesis)
        for data defined at spatially distributed sampling points
        described by a triangular mesh. The forward transformation is
        performed by matrix multiplication of the data matrix and the
        matrix with SPHARA basis functions :math:`\tilde{X} = X \cdot S`,
        with the SPHARA basis :math:`S`, the data matrix :math:`X` and the
        SPHARA coefficients matrix :math:`\tilde{X}`. In the forward
        transformation using SPHARA basic functions determined by
        discretization with FEM approach, the modified scalar product
        including the mass matrix is used
        :math:`\tilde{X} = X \cdot B \cdot S`, with the mass matrix
        :math:`B`.

        Parameters
        ----------
        coefficients : tensor, shape (m, n_points)
            A matrix containing the SPHARA coefficients. The coefficients
            are sorted column by column with increasing spatial frequency,
            starting with DC in the first column.

        Returns
        -------
        data : tensor, shape (m, n_points)
            A matrix with data to be forward transformed (analyzed) by
            SPHARA. The number of vertices of the triangular mesh is
            n_points. The order of the spatial sample points must correspond
            to that in the vertex list used to determine the SPHARA basis
            functions.

        Examples
        --------
        >>> mesh = TriMesh([[0, 1, 2]], [[1., 0, 0], [0, 2, 0], [0, 0, 3]])
        >>> st = SpharaTransform(mesh, mode='inv_euclidean')
        >>> data = torch.randn(20, 3)
        >>> coefficients = st_fem_simple.analysis(data)
        >>> recovered = st_fem_simple.synthesis(coefficients)
        >>> assert torch.allclose(data, recovered)

        """
        if self.frequencies is None or self.basis is None:
            self.construct_basis()

        # Does the number of SPHARA coefficients correspond to that of
        # the basis functions?
        if self.basis.shape[0] != coefficients.shape[1]:
            raise ValueError('Dimension mismatch')

        # compute the data from SPHARA coefficients
        data = coefficients @ self.basis.t()

        return data
