from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor as T
from torch_geometric.nn import MessagePassing


class InteractionNetworkLayer(MessagePassing):
    def __init__(self, dim: int):
        super().__init__(aggr="add")
        self.f = nn.Sequential(
            nn.Linear(dim * 3, dim),  # Assuming edge are node features are of same dimension
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.g = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x, edge_index, edge_attr) -> Tuple[T, T]:
        updated_ef = self.edge_update(x, edge_index, edge_attr)
        updated_nf = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return updated_nf, updated_ef

    def edge_update(self, x, edge_index, edge_attr):
        row, col = edge_index
        x_i, x_j = x[row], x[col]  # src and dst node features
        return self.f(torch.cat([x_i, x_j, edge_attr], dim=-1))  # Eq (1)

    # Eq (2) related
    def message(self, edge_attr):
        return edge_attr

    def update(self, aggr_msg, x):
        return self.g(torch.cat([x, aggr_msg], dim=-1))
