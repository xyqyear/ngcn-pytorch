import torch
import torch_geometric


def normalize_sparse_adjacency_matrix(adj):
    """Normalize sparse adjacency matrix with $D^{-{1 \over 2}}AD^{-{1 \over 2}}$."""
    row, col = adj._indices()
    rowsum = torch.sparse.sum(adj, dim=1).to_dense()
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    return torch.sparse_coo_tensor(
        adj._indices(),
        d_inv_sqrt[row] * adj._values() * d_inv_sqrt[col],
        adj.shape,
    )


def pyg_data_to_sparse_adj(pyg_data):
    """Convert PyG data object to torch sparse adjacency matrix and add self-loops."""
    device = pyg_data.x.device
    # can't figure out how to pass a shape on "cuda"
    pyg_data = pyg_data.to("cpu")
    # add self loops to edge_index
    edge_index, _ = torch_geometric.utils.add_self_loops(pyg_data.edge_index)
    return torch.sparse_coo_tensor(
        edge_index,
        torch.ones(edge_index.shape[1]),
        [pyg_data.x.shape[0], pyg_data.x.shape[0]],
    ).to(device)
