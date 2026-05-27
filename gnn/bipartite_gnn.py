"""Bipartite Graph Neural Network for column classification.

Implements Algorithm 1 from EBSCO paper (Section 3.3.2):

  Bipartite graph: column nodes V + constraint nodes C
  Edges: (v, c) if column v contributes to constraint c

  For k = 1..K:
    Phase 1: constraint update  h_c = ψ_C([h_c, Σ φ_C(h_c, h_v)])
    Phase 2: column update      h_v = ψ_V([h_v, Σ φ_V(h_v, h_c)])

  Output: y_v = out(h_v^K)

Architecture (Table 2):
  K = 1 (message-passing iterations)
  φ, ψ: 2-layer MLP, 32 hidden units each, ReLU
  out: 3-layer MLP (32→32→1) + Sigmoid
  Loss: weighted BCE (10:1 pos:neg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_sum(src, index, dim_size):
    """Scatter-add operation: sum src values by index.

    Equivalent to torch_scatter.scatter_sum or PyG's scatter(..., reduce='sum').

    Args:
        src: (N, D) tensor of messages
        index: (N,) tensor of target indices
        dim_size: int, number of target nodes
    Returns:
        (dim_size, D) tensor of aggregated messages
    """
    out = src.new_zeros(dim_size, src.size(1))
    out.index_add_(0, index, src)
    return out


class MLP(nn.Module):
    """Configurable multi-layer perceptron."""

    def __init__(self, in_dim, hidden_dims, out_dim, final_relu=False):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or final_relu:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BipartiteGNN(nn.Module):
    """Bipartite GNN for column selection.

    Args:
        col_feat_dim: dimension of column node features (d=12)
        constr_feat_dim: dimension of constraint node features (p=2)
        hidden_dim: hidden dimension for φ, ψ networks (32)
        num_iterations: message-passing rounds K (1)
    """

    def __init__(self, col_feat_dim=12, constr_feat_dim=2, hidden_dim=32, num_iterations=1):
        super().__init__()
        self.col_feat_dim = col_feat_dim
        self.constr_feat_dim = constr_feat_dim
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations

        # φ_C: takes (h_c(prev), h_v(prev)) → aggregated per constraint
        self.phi_C = MLP(constr_feat_dim + col_feat_dim, [hidden_dim], hidden_dim)

        # ψ_C: takes [h_c(prev), a_c] → updated constraint representation
        self.psi_C = MLP(constr_feat_dim + hidden_dim, [hidden_dim], hidden_dim)

        # φ_V: takes (h_v(prev), h_c(updated)) → aggregated per column
        self.phi_V = MLP(col_feat_dim + hidden_dim, [hidden_dim], hidden_dim)

        # ψ_V: takes [h_v(prev), a_v] → updated column representation
        self.psi_V = MLP(col_feat_dim + hidden_dim, [hidden_dim], hidden_dim)

        # Output: 3-layer MLP with sigmoid
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, col_features, constr_features, edge_index):
        """Forward pass.

        Args:
            col_features: (num_cols, col_feat_dim) column node features
            constr_features: (num_constraints, constr_feat_dim) constraint node features
            edge_index: (2, num_edges) bipartite edges [col_idx, constr_idx]

        Returns:
            logits: (num_cols,) raw logits (before sigmoid) for each column
        """
        h_v = col_features  # (n_cols, d)
        h_c = constr_features  # (n_constraints, p)
        col_idx = edge_index[0]  # (E,)
        constr_idx = edge_index[1]  # (E,)
        n_cols = h_v.size(0)
        n_constraints = h_c.size(0)

        for _ in range(self.num_iterations):
            # ---- Phase 1: Constraint update ----
            # Collect (h_c, h_v) pairs for each edge
            h_c_per_edge = h_c[constr_idx]  # (E, p)
            h_v_per_edge = h_v[col_idx]     # (E, d)
            phi_input = torch.cat([h_c_per_edge, h_v_per_edge], dim=1)  # (E, p+d)

            # φ_C on each edge, then scatter-sum to constraints
            msg_to_c = self.phi_C(phi_input)  # (E, hidden)
            a_c = scatter_sum(msg_to_c, constr_idx, n_constraints)  # (n_constraints, hidden)

            # ψ_C: update constraint representations
            h_c = self.psi_C(torch.cat([h_c, a_c], dim=1))  # (n_constraints, hidden)

            # ---- Phase 2: Column update ----
            h_c_per_edge = h_c[constr_idx]  # (E, hidden)
            h_v_per_edge = h_v[col_idx]     # (E, d)
            phi_input = torch.cat([h_v_per_edge, h_c_per_edge], dim=1)  # (E, d+hidden)

            msg_to_v = self.phi_V(phi_input)  # (E, hidden)
            a_v = scatter_sum(msg_to_v, col_idx, n_cols)  # (n_cols, hidden)

            h_v = self.psi_V(torch.cat([h_v, a_v], dim=1))  # (n_cols, hidden)

            # After first iteration, h_c and h_v are both hidden_dim
            # For K>1: update input dimensions (not needed for K=1)
            if _ == 0 and self.num_iterations > 1:
                # Rebuild φ_C, ψ_C, φ_V, ψ_V for new dimensions
                # Not needed for K=1, skip for simplicity
                pass

        # Output: sigmoid probabilities for each column
        logits = self.out(h_v).squeeze(-1)  # (n_cols,)
        return logits
