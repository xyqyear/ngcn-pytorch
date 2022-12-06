import torch
import torch.nn.functional as F

from utils import normalize_sparse_adjacency_matrix


class NGCNCell(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, power, dropout):
        super(NGCNCell, self).__init__()
        self.power = power
        self.hidden_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, normed_sparse_adj):
        for _ in range(self.power):
            x = torch.sparse.mm(normed_sparse_adj, x)
        x = self.hidden_layer(x)

        for _ in range(self.power):
            x = torch.sparse.mm(normed_sparse_adj, x)
        x = self.output_layer(x)

        return x


class NGCN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        max_hops=3,
        replicas=3,
        gcn_hidden_dim=10,
        output_layer="concat",
        input_dropout=0.8,
        layer_dropout=0.4,
    ):
        """
        input_dim: dimension of input features
        output_dim: number of classes
        max_hops: number of hops
        replicas: number of replicas
        gcn_hidden_dim: hidden dimension for the first GCN layers
        output_layer: 'wsum' or 'concat'
        dropout: dropout rate
        """
        super(NGCN, self).__init__()
        self.output_layer = output_layer
        self.input_dropout = input_dropout

        # we will have replicas * (max_hops + 1) cells
        self.replicas = torch.nn.ModuleList()
        for r in range(replicas):
            # then for each replica, we will have max_hops + 1 cells, each cell with different powers
            self.replicas.append(torch.nn.ModuleList())
            for h in range(max_hops + 1):
                self.replicas[r].append(
                    NGCNCell(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        hidden_dim=gcn_hidden_dim,
                        power=h,
                        dropout=layer_dropout,
                    )
                )

        match output_layer:
            case "wsum":
                # if we are using wsum, we need to learn a weight for each replica and power
                self.weights = torch.nn.Parameter(torch.Tensor(replicas, max_hops + 1))
                torch.nn.init.uniform_(self.weights)
            case "concat":
                self.output_linear = torch.nn.Linear(
                    replicas * (max_hops + 1) * output_dim, output_dim
                )
            case _:
                raise ValueError("output_layer must be 'wsum' or 'concat'")

    def forward(self, x, sparse_adj):
        """
        x: [N, input_dim]
        sparse_adj: [N, N] sparse tensor
        """
        x = F.dropout(x, self.input_dropout, training=self.training)
        x = F.normalize(x, p=2, dim=1)

        normed_sparse_adj = normalize_sparse_adjacency_matrix(sparse_adj)
        cell_outputs = []
        for r in range(len(self.replicas)):
            for h in range(len(self.replicas[r])):
                cell_outputs.append(self.replicas[r][h](x, normed_sparse_adj))
        cell_outputs = torch.stack(cell_outputs, dim=1)

        match self.output_layer:
            case "wsum":
                softmax_weights = F.softmax(self.weights.view(-1), dim=0)
                output = torch.sum(cell_outputs * softmax_weights.unsqueeze(-1), dim=1)
            case "concat":
                output = cell_outputs.view(cell_outputs.shape[0], -1)
                output = F.relu(output)
                output = self.output_linear(output)

        return output, cell_outputs
