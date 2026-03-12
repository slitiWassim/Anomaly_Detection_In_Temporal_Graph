import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_mean



"""

model Results : 88% -> 92%


"""





class EdgeEncoder(nn.Module):
    def __init__(self, edge_dim, hidden_dim):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, edge_attr):
        return self.lin(edge_attr)



class TemporalMemory(nn.Module):
    def __init__(self, num_nodes, dim, momentum=0.95):
        super().__init__()
        self.register_buffer("memory", torch.zeros(num_nodes, dim))
        self.momentum = momentum

    def read(self, idx):
        return self.memory[idx]

    def write(self, idx, emb):
        with torch.no_grad():
            self.memory[idx] = (
                self.momentum * self.memory[idx] +
                (1 - self.momentum) * emb.detach()
            )


class EdgeAwareQKV(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.q = nn.Linear(dim, dim)
        self.k_node = nn.Linear(dim, dim)
        self.v_node = nn.Linear(dim, dim)

        self.k_edge = nn.Linear(dim, dim)
        self.v_edge = nn.Linear(dim, dim)

    def split(self, x):
        return x.view(-1, self.heads, self.head_dim)

    def forward(self, x, edge_emb):
        Q = self.split(self.q(x))
        K_node = self.split(self.k_node(x))
        V_node = self.split(self.v_node(x))

        K_edge = self.split(self.k_edge(edge_emb))
        V_edge = self.split(self.v_edge(edge_emb))

        return Q, K_node, V_node, K_edge, V_edge
    


class EdgeAwareTrust(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, dim),
            nn.ReLU(),
            nn.Linear(dim, heads),
            nn.Sigmoid()
        )

    def forward(self, delta_hist, dev_nei, volatility, edge_norm):
        x = torch.stack(
            [delta_hist, dev_nei, volatility, edge_norm],
            dim=-1
        )
        return self.mlp(x).unsqueeze(-1)
    


class EdgeAwareTemporalAttention(MessagePassing):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__(aggr="add", node_dim=0)

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = EdgeAwareQKV(dim, heads)
        self.trust_net = EdgeAwareTrust(dim, heads)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, edge_index, edge_emb, memory):
        Q, K_node, V_node, K_edge, V_edge = self.qkv(x, edge_emb)

        out = self.propagate(
            edge_index,
            Q=Q,
            K_node=K_node,
            V_node=V_node,
            K_edge=K_edge,
            V_edge=V_edge,
            memory=memory
        )

        out = out.view(-1, self.dim)
        return self.norm(x + out)    
    
    def message(
        self,
        Q_i,
        K_node_j,
        V_node_j,
        K_edge,
        V_edge,
        memory,
        index
    ):
        # ---------- ATTENTION ----------
        K = K_node_j + K_edge
        attn_raw = (Q_i * K).sum(dim=-1) * self.scale

        # ---------- TRUST SIGNALS ----------
        mem_j = memory.read(index)

        delta_hist = torch.norm(
            K_node_j.reshape(K_node_j.size(0), -1) - mem_j,
            dim=-1
        )

        neigh_mean = scatter_mean(
            K_node_j.reshape(K_node_j.size(0), -1),
            index,
            dim=0
        )[index]

        dev_nei = torch.norm(
            K_node_j.reshape(K_node_j.size(0), -1) - neigh_mean,
            dim=-1
        )

        volatility = torch.abs(attn_raw).mean(dim=-1)
        edge_norm = torch.norm(
            K_edge.reshape(K_edge.size(0), -1),
            dim=-1
        )

        trust = self.trust_net(
            delta_hist,
            dev_nei,
            volatility,
            edge_norm
        )

        # ---------- TRUST-AWARE SOFTMAX ----------
        attn = softmax(attn_raw * trust.squeeze(-1), index)
        attn = self.dropout(attn)

        # ---------- MESSAGE ----------
        V = V_node_j + V_edge
        out = V * attn.unsqueeze(-1) * trust

        return out
    



class EdgeAwareTemporalGNN(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_dim,
        edge_dim,
        hidden_dim,
        num_layers=2,
        heads=4
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_enc = EdgeEncoder(edge_dim, hidden_dim)
        self.memory = TemporalMemory(num_nodes, hidden_dim)

        self.layers = nn.ModuleList([
            EdgeAwareTemporalAttention(hidden_dim, heads)
            for _ in range(num_layers)
        ])

    def forward(self, batch):
        x = self.input_proj(batch.x)
        edge_emb = self.edge_enc(batch.msg)

        for layer in self.layers:
            x = layer(x, batch.edge_index, edge_emb, self.memory)

        self.memory.write(
            torch.arange(x.size(0), device=x.device),
            x
        )

        return x