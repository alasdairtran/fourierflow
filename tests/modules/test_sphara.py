import torch

from fourierflow.modules.sphara.trimesh import TriMesh, get_triangle_area


def test_area_of_standard_simplex():
    vertex_coords = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    area = get_triangle_area(vertex_coords)
    assert area == 0.8660254037844386


def test_area_of_two_R3_triangles():
    vertex_coords = torch.Tensor([
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[7.8, 1.5, 10.5], [7.8, 1.5, 20.5], [12.8, 1.5, 10.5]]
    ])
    area = get_triangle_area(vertex_coords)
    targets = torch.Tensor([0.5, 25])
    assert torch.allclose(area, targets)


def test_area_of_two_R2_triangles():
    vertex_coords = torch.Tensor([
        [[0, 0], [1, 0], [0, 1]],
        [[7.8, 10.5], [7.8, 20.5], [12.8, 10.5]]
    ])
    area = get_triangle_area(vertex_coords)
    targets = torch.Tensor([0.5, 25])
    assert torch.allclose(area, targets)


def test_trimesh_normal_mass_matrix():
    triangles = torch.tensor([[0, 1, 2]])
    vertices = torch.tensor([[1.0, 0, 0], [0, 2, 0], [0, 0, 3]])
    mesh = TriMesh(triangles, vertices)
    mass = mesh.get_mass_matrix(mode='normal')

    targets = torch.tensor([[0.58333333,  0.29166667,  0.29166667],
                            [0.29166667,  0.58333333,  0.29166667],
                            [0.29166667,  0.29166667,  0.58333333]])

    assert torch.allclose(mass.to_dense(), targets)


def test_trimesh_half_cotangent_weight_matrix():
    triangles = torch.tensor([[0, 1, 2]])
    vertices = torch.tensor([[1.0, 0, 0], [0, 2, 0], [0, 0, 3]])
    mesh = TriMesh(triangles, vertices)
    weight = mesh.get_weight_matrix(mode='half_cotangent')

    targets = torch.tensor([[0., 0.64285714, 0.28571429],
                            [0.64285714, 0., 0.07142857],
                            [0.28571429, 0.07142857, 0.]])

    assert torch.allclose(weight.to_dense(), targets)


def test_trimesh_inv_euclidean_weight_matrix():
    triangles = torch.tensor([[0, 1, 2]])
    vertices = torch.tensor([[1.0, 0, 0], [0, 2, 0], [0, 0, 3]])
    mesh = TriMesh(triangles, vertices)
    weight = mesh.get_weight_matrix(mode='inv_euclidean')

    targets = torch.tensor([[0., 0.4472136, 0.31622777],
                            [0.4472136, 0., 0.2773501],
                            [0.31622777, 0.2773501, 0.]])

    assert torch.allclose(weight.to_dense(), targets)


def test_trimesh_unit_weight_matrix():
    triangles = torch.tensor([[0, 1, 2]])
    vertices = torch.tensor([[1.0, 0, 0], [0, 2, 0], [0, 0, 3]])
    mesh = TriMesh(triangles, vertices)
    weight = mesh.get_weight_matrix(mode='unit')

    targets = torch.tensor([[0., 1., 1.],
                            [1., 0., 1.],
                            [1., 1., 0.]])

    assert torch.allclose(weight.to_dense(), targets)


def test_trimesh_stiffness_matrix():
    triangles = torch.tensor([[0, 1, 2]])
    vertices = torch.tensor([[1.0, 0, 0], [0, 2, 0], [0, 0, 3]])
    mesh = TriMesh(triangles, vertices)
    stiffness = mesh.get_stiffness_matrix()

    targets = torch.tensor([[-0.92857143,  0.64285714,  0.28571429],
                            [0.64285714, -0.71428571,  0.07142857],
                            [0.28571429,  0.07142857, -0.35714286]])

    assert torch.allclose(stiffness.to_dense(), targets)
