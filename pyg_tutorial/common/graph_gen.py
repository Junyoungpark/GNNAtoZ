import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph


def generate_random_graph(
    num_node: int, p_edges: float = 0.25, node_feat_dim: int = 5, edge_feat_dim: int = 5
):
    x = torch.randn(size=(num_node, node_feat_dim))
    # assume directional and potentially self-loop
    num_edges = int(num_node * num_node * p_edges)

    edge_index = torch.stack(
        [torch.arange(num_node).repeat(num_node), torch.arange(num_node).repeat(num_node)], dim=0
    )
    edge_index = edge_index[:, torch.randperm(edge_index.shape[1])]
    edge_index = edge_index[:, :num_edges]

    edge_attr = torch.randn(size=(num_edges, edge_feat_dim))

    dummy_attr = torch.randn(size=(num_node, 1))

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, dummy_attr=dummy_attr)


def generate_er_graph(num_nodes: int, edge_prob: float, feat_dim: int = 16):
    edge_idx = erdos_renyi_graph(num_nodes=num_nodes, edge_prob=edge_prob)
    x = torch.randn(num_nodes, feat_dim)
    y = (x.sum() / num_nodes).view(1, 1)
    g = Data(x=x, edge_index=edge_idx, y=y)
    return g
