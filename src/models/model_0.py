import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_mean


"""

model 0 : Results  90 : +1  -0.5 


"""



class RelativeTimeEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(dim))
        self.decay = nn.Parameter(torch.tensor(0.1))

    def forward(self, t_diff):
        t = t_diff.float().unsqueeze(-1)
        return torch.sin(t * self.freq) * torch.exp(-torch.abs(t) * self.decay)


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


class MessageEvolution(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, msg, prev):
        return self.gru(msg, prev)



class TrustAwareTemporalAttention(MessagePassing):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__(aggr='add', node_dim=0)

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.lin_q = nn.Linear(dim, dim)
        self.lin_k = nn.Linear(dim, dim)
        self.lin_v = nn.Linear(dim, dim)
        self.lin_t = nn.Linear(dim, dim)

        # Trust MLP (temporal + neighborhood + similarity)
        self.trust_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, heads)
        )

        self.msg_evo = MessageEvolution(dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, edge_index, t_emb, memory):
        q = self.lin_q(x).view(-1, self.heads, self.head_dim)
        k = self.lin_k(x).view(-1, self.heads, self.head_dim)
        v = self.lin_v(x).view(-1, self.heads, self.head_dim)

        out = self.propagate(
            edge_index,
            q=q, k=k, v=v,
            t_emb=t_emb,
            memory=memory,
            x=x
        )

        out = out.view(-1, self.dim)
        return self.norm(x + out)

        
    def message(self, q_i, k_j, v_j, t_emb, index, memory, x):
        # time embedding
        t_h = self.lin_t(t_emb).view(-1, self.heads, self.head_dim)

        # attention score
        attn = (q_i * (k_j + t_h)).sum(dim=-1) * self.scale

        # -------- TRUST SIGNALS -------- #

        # (1) temporal inconsistency
        mem_j = memory.read(index)
        temporal_diff = torch.norm(k_j.view(k_j.size(0), -1) - mem_j, dim=-1)

        # (2) neighborhood deviation
        neigh_mean = scatter_mean(
            k_j.view(k_j.size(0), -1),
            index,
            dim=0
        )
        dev = torch.norm(
            k_j.view(k_j.size(0), -1) - neigh_mean[index],
            dim=-1
        )

        # (3) similarity
        sim = (q_i * k_j).sum(dim=-1).mean(dim=-1)

        trust_input = torch.stack([temporal_diff, dev, sim], dim=-1)
        trust = torch.sigmoid(self.trust_mlp(trust_input))
        trust = trust.unsqueeze(-1)

        # trust-aware attention normalization
        attn = softmax(attn * trust.squeeze(-1), index)
        attn = self.dropout(attn)

        # message evolution
        v_flat = v_j.view(v_j.size(0), -1)
        evolved_msg = self.msg_evo(v_flat, mem_j)

        out = evolved_msg.view(-1, self.heads, self.head_dim)
        out = out * attn.unsqueeze(-1) * trust

        return out




class TrustAwareTemporalGNN(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_dim,
        hidden_dim,
        num_layers=2,
        heads=4
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.time_enc = RelativeTimeEncoding(hidden_dim)
        self.memory = TemporalMemory(num_nodes, hidden_dim)

        self.layers = nn.ModuleList([
            TrustAwareTemporalAttention(hidden_dim, heads)
            for _ in range(num_layers)
        ])

    def forward(self, batch):
        x = self.input_proj(batch.x)

        if hasattr(batch, 'edge_time_diff'):
            t_emb = self.time_enc(batch.edge_time_diff)
        else:
            t_emb = torch.zeros(
                batch.edge_index.size(1),
                x.size(1),
                device=x.device
            )

        for layer in self.layers:
            x = layer(x, batch.edge_index, t_emb, self.memory)

        # update memory
        self.memory.write(
            torch.arange(x.size(0), device=x.device),
            x
        )

        return x
