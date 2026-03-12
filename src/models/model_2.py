import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_mean



"""

model Results : 89% -> 91%


"""




class AdvancedSimilarityTrust(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        self.heads = heads

    def forward(self, q_i, k_j, mem_i, mem_j):
        cos_sim = F.cosine_similarity(q_i, k_j, dim=-1)
        drift = (q_i - mem_i).norm(dim=-1)
        deviation = (k_j - mem_j).norm(dim=-1)

        features = torch.stack([cos_sim, drift, deviation], dim=-1)
        trust = self.mlp(features).squeeze(-1)
        return torch.sigmoid(trust)
    


class EnhancedTemporalTrustAttention(MessagePassing):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__(aggr="add", node_dim=0)

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Node projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Edge + Time projections
        self.edge_proj = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(dim, dim)

        self.trust_net = AdvancedSimilarityTrust(heads)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.residual_gate = nn.Linear(dim, 1)

    def split(self, x):
        return x.view(-1, self.heads, self.head_dim)

    def forward(self, x, edge_index, edge_emb, time_emb, memory):

        Q = self.split(self.q_proj(x))
        K = self.split(self.k_proj(x))
        V = self.split(self.v_proj(x))

        mem = memory.read(torch.arange(x.size(0), device=x.device))
        mem = self.split(mem)

        edge_emb = self.split(self.edge_proj(edge_emb))
        time_emb = self.split(self.time_proj(time_emb))

        out = self.propagate(
            edge_index,
            Q=Q,
            K=K,
            V=V,
            mem=mem,
            edge_emb=edge_emb,
            time_emb=time_emb
        )

        out = out.view(-1, self.dim)

        beta = torch.sigmoid(self.residual_gate(x))
        return self.norm(beta * x + (1 - beta) * out)
    

    def message(self, Q_i, K_j, V_j,
                mem_i, mem_j,
                edge_emb, time_emb,
                index):

        # Add edge + time to keys
        K_j = K_j + edge_emb + time_emb

        # Attention
        attn = (Q_i * K_j).sum(dim=-1) * self.scale

        # Trust modulation
        trust = self.trust_net(Q_i, K_j, mem_i, mem_j)

        attn = attn * trust

        attn = softmax(attn, index)
        attn = self.dropout(attn)

        return V_j * attn.unsqueeze(-1)
    


class TemporalMemory(nn.Module):
    def __init__(self, num_nodes, dim, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("memory", torch.zeros(num_nodes, dim))

    def read(self, idx):
        return self.memory[idx]

    def write(self, idx, values):
        self.memory[idx] = (
            self.momentum * self.memory[idx]
            + (1 - self.momentum) * values.detach()
        )


class EnhancedTrustTemporalGNN(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_dim,
        edge_dim,
        hidden_dim,
        num_layers=2,
        heads=4,
        memory_momentum=0.9
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_enc = nn.Linear(edge_dim, hidden_dim)
        self.time_enc = nn.Linear(1, hidden_dim)

        self.memory = TemporalMemory(
            num_nodes,
            hidden_dim,
            momentum=memory_momentum
        )

        self.layers = nn.ModuleList([
            EnhancedTemporalTrustAttention(
                hidden_dim,
                heads=heads
            )
            for _ in range(num_layers)
        ])

    def forward(self, batch):

        x = self.input_proj(batch.x)

        edge_emb = self.edge_enc(batch.msg)

        # make sure time is shape [num_edges,1]
        t = batch.t.view(-1, 1).float()
        time_emb = self.time_enc(t)

        for layer in self.layers:
            x = layer(
                x,
                batch.edge_index,
                edge_emb,
                time_emb,
                self.memory
            )

        self.memory.write(
            torch.arange(x.size(0), device=x.device),
            x
        )

        return x