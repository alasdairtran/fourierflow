import torch
import torch.nn as nn

from fourierflow.registries import Module


@Module.register('mesh_fourier')
class MeshFourier(Module):
    def __init__(self, n_layers: int, input_size: int, hidden_size: int,
                 output_size: int, n_nodes: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.in_proj = nn.Linear(input_size, hidden_size)

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            layer = MeshFourierLayer(hidden_size=hidden_size, n_nodes=n_nodes)
            self.layers.append(layer)

        self.out_proj = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, mass, basis):
        # mass.shape == [n_nodes, n_nodes]
        # basis.shape == [n_nodes, n_basis]

        x = self.in_proj(inputs)
        # x.shape == [n_nodes, hidden_size]

        for layer in self.layers:
            x = layer(x, mass, basis) + x

        preds = self.out_proj(x)
        # x.shape == [n_nodes, output_size]

        return {'preds': preds}


class MeshFourierLayer(nn.Module):
    def __init__(self, hidden_size: int, n_nodes: int):
        super().__init__()

        self.weight = nn.Parameter(
            torch.FloatTensor(n_nodes, hidden_size, hidden_size))
        nn.init.xavier_normal_(self.weight)

        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size))

    def forward(self, x, mass, basis):
        res = x

        # x = torch.einsum('nh,nm->mh', x, mass)
        # # x.shape == [n_nodes, hidden_size]

        # x = torch.einsum('mh,mb->bh', x, basis)
        # # x.shape == [n_basis, hidden_size]

        # x = torch.einsum('bi,bio->bo', x, self.weight)
        # # x.shape == [n_basis, hidden_size]

        # x = torch.einsum('bh,mb->mh', x, basis)
        # # x.shape == [n_nodes, hidden_size]

        x = self.feedforward(x) + res

        return x
