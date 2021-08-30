import torch

from fourierflow.modules.sphara.trimesh import get_triangle_area


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
