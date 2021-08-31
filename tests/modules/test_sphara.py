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
