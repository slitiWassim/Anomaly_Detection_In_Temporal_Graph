
"""
Enhanced Temporal Trust GNN — Recurrent + Frequency-Aware Edition
==================================================================
Key architectural upgrades over the 87% baseline:

  1.  Learnable Time Encoding  — Bochner / random Fourier features replace
      scalar linear encoding; captures periodicity and multi-scale frequency.

  2.  Recurrent Node Memory (GRU-based)  — each node maintains a hidden state
      that is updated every time it appears, giving a true sequential view of
      its behavioral history.

  3.  Interaction Frequency Tracker  — counts how often each node interacts
      and decays the count with an exponential half-life, producing a
      "burstiness" signal.

  4.  Neighborhood Evolution Bank  — stores a rolling window of neighbor
      embeddings per node; the trust score is computed over HOW a node's
      neighborhood has changed, not just its current state.

  5.  Temporal Cross-Attention Trust  — trust between i and j is now based on
      comparing the EVOLUTION TRAJECTORY of j's neighborhood against node i's
      recurrent history, using cross-attention over time steps.

  6.  Fourier Time-Aware Positional Bias  — attention logits get an additive
      bias from the Fourier similarity between edge timestamps and node's
      last-seen time (recency-sensitive attention).

  7.  Recurrent-Augmented Message Aggregation  — aggregated messages are
      fused with the node's GRU hidden state before the output projection,
      so the GNN output reflects both local structure AND temporal history.
"""



"""
Recurrent Evolution GNN — Shape-Fixed Version
==============================================
Fixes:
  - var_i/var_j shape mismatch in TemporalEvolutionTrust (was heads×head_dim, now flat dim)
  - recency_bias shape mismatch (was indexed by N when size is E)
  - evo_vals/evo_times propagation through MessagePassing (window dim causes issues, flattened)
  - mem_vars_split fallback always produces correct (N, dim) shape
  - time_per_node used correctly as AdaLN condition (projected to hidden_dim)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


"""

Results : 87 % 

"""



# ═══════════════════════════════════════════════════════════════════════════════
# 1. Fourier Time Encoding
# ═══════════════════════════════════════════════════════════════════════════════

class FourierTimeEncoding(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        assert out_dim % 2 == 0
        self.k  = out_dim // 2
        self.W  = nn.Parameter(torch.randn(1, self.k) * 0.02)
        self.b  = nn.Parameter(torch.zeros(1, self.k))
        self.w0 = nn.Parameter(torch.ones(1, 1))
        self.b0 = nn.Parameter(torch.zeros(1, 1))
        self.out_proj = nn.Linear(self.k + 1, out_dim)

    def forward(self, t):
        t        = t.float().view(-1, 1)
        trend    = t * self.w0 + self.b0
        periodic = torch.sin(t * self.W + self.b)
        return self.out_proj(torch.cat([trend, periodic], dim=-1))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Interaction Frequency Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class FrequencyTracker(nn.Module):
    def __init__(self, num_nodes, decay=0.95):
        super().__init__()
        self.decay = decay
        self.register_buffer("count",     torch.zeros(num_nodes))
        self.register_buffer("last_time", torch.zeros(num_nodes))

    def read(self, idx):
        return self.count[idx]

    @torch.no_grad()
    def update(self, idx, t):
        elapsed            = (t.float() - self.last_time[idx]).clamp(min=0)
        decay              = self.decay ** elapsed
        self.count[idx]    = decay * self.count[idx] + 1.0
        self.last_time[idx] = t.float()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Recurrent Node Memory (GRU-based)
# ═══════════════════════════════════════════════════════════════════════════════

class RecurrentNodeMemory(nn.Module):
    def __init__(self, num_nodes, dim, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.dim      = dim
        self.gru_cell = nn.GRUCell(dim, dim)
        self.register_buffer("hidden",   torch.zeros(num_nodes, dim))
        self.register_buffer("variance", torch.ones(num_nodes, dim))

    def read(self, idx):
        return self.hidden[idx], self.variance[idx]          # (B,D), (B,D)

    def get_hidden(self, idx):
        return self.hidden[idx]

    @torch.no_grad()
    def write(self, idx, new_repr):
        h_prev = self.hidden[idx]
        with torch.enable_grad():
            h_new = self.gru_cell(new_repr.detach(), h_prev)
        delta              = h_new.detach() - h_prev
        self.variance[idx] = (self.momentum * self.variance[idx]
                              + (1 - self.momentum) * delta ** 2)
        self.hidden[idx]   = h_new.detach()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Neighborhood Evolution Bank
# ═══════════════════════════════════════════════════════════════════════════════

class NeighborhoodEvolutionBank(nn.Module):
    def __init__(self, num_nodes, dim, window=8):
        super().__init__()
        self.window = window
        self.dim    = dim
        self.register_buffer("bank",       torch.zeros(num_nodes, window, dim))
        self.register_buffer("timestamps", torch.zeros(num_nodes, window))
        self.register_buffer("ptr",        torch.zeros(num_nodes, dtype=torch.long))

    def read(self, idx):
        return self.bank[idx], self.timestamps[idx]          # (B,W,D), (B,W)

    @torch.no_grad()
    def write(self, idx, neighbor_repr, t):
        for bi in range(idx.size(0)):
            p = int(self.ptr[idx[bi]].item()) % self.window
            self.bank[idx[bi], p]       = neighbor_repr[bi].detach()
            self.timestamps[idx[bi], p] = t[bi].float()
            self.ptr[idx[bi]]          += 1


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Temporal Evolution Trust  ← SHAPE BUG FIXED HERE
#    var_i / var_j arrive as (E, dim) flat tensors — NOT split into heads
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalEvolutionTrust(nn.Module):
    def __init__(self, dim, num_heads=4, window=8):
        super().__init__()
        # Use dim directly — no head_dim split inside trust
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.window    = window

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.time_decay      = nn.Parameter(torch.ones(1) * 0.1)
        self.uncertainty_gate = nn.Linear(dim, 1)
        self.out             = nn.Linear(dim, 1)
        self.scale           = self.head_dim ** -0.5

    def forward(self, h_i, h_j, evol_j, evol_t_j, var_i, var_j,
                current_t, eps=1e-6):
        """
        h_i:      (E, dim)
        h_j:      (E, dim)
        evol_j:   (E, W, dim)
        evol_t_j: (E, W)
        var_i:    (E, dim)   ← flat, not split into heads
        var_j:    (E, dim)
        current_t:(E,)
        """
        E, W, D = evol_j.shape
        nh, hd  = self.num_heads, self.head_dim

        Q = self.q_proj(h_i).view(E, nh, hd)                        # (E, nh, hd)

        evol_flat = evol_j.reshape(E * W, D)
        K = self.k_proj(evol_flat).view(E, W, nh, hd)               # (E, W, nh, hd)
        V = self.v_proj(evol_flat).view(E, W, nh, hd)

        # Attention: Q over evolution window
        Q_exp = Q.unsqueeze(2)                                       # (E, nh, 1, hd)
        K_t   = K.permute(0, 2, 1, 3)                               # (E, nh, W, hd)
        attn  = (Q_exp * K_t).sum(-1) * self.scale                  # (E, nh, W)

        # Time-decay bias
        dt        = (current_t.unsqueeze(1) - evol_t_j).clamp(min=0)   # (E, W)
        time_bias = -self.time_decay.abs() * dt                         # (E, W)
        attn      = attn + time_bias.unsqueeze(1)

        # Uncertainty gate over evolution window
        unc_gate = torch.sigmoid(
            self.uncertainty_gate(evol_j).squeeze(-1))               # (E, W)
        attn = attn * unc_gate.unsqueeze(1)

        attn   = F.softmax(attn, dim=-1)                             # (E, nh, W)
        V_t    = V.permute(0, 2, 1, 3)                              # (E, nh, W, hd)
        pooled = (attn.unsqueeze(-1) * V_t).sum(2)                  # (E, nh, hd)
        pooled = pooled.reshape(E, D)                                # (E, dim)

        # ── Confidence from flat var ─────────────────────────────────────
        conf_i = 1.0 / (1.0 + var_i.mean(-1).sqrt().clamp(min=eps)) # (E,)
        conf_j = 1.0 / (1.0 + var_j.mean(-1).sqrt().clamp(min=eps)) # (E,)

        trust_logit = self.out(pooled).squeeze(-1)                   # (E,)
        return torch.sigmoid(trust_logit * conf_i * conf_j)          # (E,)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Rotary Embedding
# ═══════════════════════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, t):
        freqs    = torch.outer(t.float(), self.inv_freq)
        cos      = freqs.cos().unsqueeze(1)
        sin      = freqs.sin().unsqueeze(1)
        x1, x2  = x[..., ::2], x[..., 1::2]
        x_rot    = torch.stack([-x2, x1], dim=-1).flatten(-2)
        cos_full = cos.expand_as(x1).repeat_interleave(2, dim=-1)
        sin_full = sin.expand_as(x1).repeat_interleave(2, dim=-1)
        return x * cos_full + x_rot * sin_full


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Adaptive LayerNorm
# ═══════════════════════════════════════════════════════════════════════════════

class AdaLayerNorm(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 2))

    def forward(self, x, cond):
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MoE Message Router
# ═══════════════════════════════════════════════════════════════════════════════

class MoEMessageRouter(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.top_k   = top_k
        self.gate    = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(dim * 2, dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, msg):
        logits              = self.gate(msg)
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        weights             = F.softmax(topk_vals, dim=-1)
        out = torch.zeros_like(msg)
        for ki in range(self.top_k):
            idx = topk_idx[:, ki]
            for ei, expert in enumerate(self.experts):
                mask = (idx == ei)
                if mask.any():
                    out[mask] += weights[mask, ki].unsqueeze(-1) * expert(msg[mask])
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Core Attention Layer  ← ALL SHAPE BUGS FIXED
# ═══════════════════════════════════════════════════════════════════════════════

class RecurrentEvolutionAttention(MessagePassing):
    def __init__(self, dim, heads=8, kv_heads=2, dropout=0.1,
                 num_experts=4, moe_top_k=2, evo_window=8):
        assert heads % kv_heads == 0
        super().__init__(aggr="add", node_dim=0)

        self.dim        = dim
        self.heads      = heads
        self.kv_heads   = kv_heads
        self.groups     = heads // kv_heads
        self.head_dim   = dim // heads
        self.evo_window = evo_window

        self.q_proj    = nn.Linear(dim, dim)
        self.k_proj    = nn.Linear(dim, kv_heads * self.head_dim)
        self.v_proj    = nn.Linear(dim, kv_heads * self.head_dim)
        self.o_proj    = nn.Linear(dim, dim)
        self.edge_proj = nn.Linear(dim, kv_heads * self.head_dim)

        self.rope = RotaryEmbedding(self.head_dim)

        # Per-head temperature with burstiness modulation
        self.log_temp_base   = nn.Parameter(torch.zeros(heads))
        self.burst_temp_gate = nn.Linear(1, heads)

        # Trust: receives flat (E, dim) vars — no head split
        self.trust_net = TemporalEvolutionTrust(dim, num_heads=4, window=evo_window)

        # Recurrent fusion after aggregation
        self.recurrent_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))

        # Recency bias: projects time_emb (dim) → heads
        self.recency_bias_proj = nn.Linear(dim, heads)

        self.moe      = MoEMessageRouter(dim, num_experts, moe_top_k, dropout)
        self.dropout  = nn.Dropout(dropout)
        self.ada_norm = AdaLayerNorm(dim, dim)
        self.gate_lin = nn.Linear(dim, 1)

        self.trust_cache = None

    def split_q(self, x):   return x.view(-1, self.heads, self.head_dim)
    def split_kv(self, x):  return x.view(-1, self.kv_heads, self.head_dim)
    def expand_kv(self, x): return x.repeat_interleave(self.groups, dim=1)

    def forward(self, x, edge_index, edge_emb, time_emb,
                node_last_time, node_burst, memory, evo_bank):
        N      = x.size(0)
        E_num  = edge_emb.size(0)
        device = x.device

        Q = self.split_q(self.q_proj(x))
        K = self.expand_kv(self.split_kv(self.k_proj(x)))
        V = self.expand_kv(self.split_kv(self.v_proj(x)))

        Q = self.rope(Q, node_last_time)
        K = self.rope(K, node_last_time)

        # Per-node temperature: (N, heads)
        temp = (self.log_temp_base.unsqueeze(0)
                + self.burst_temp_gate(node_burst)).exp()

        # Memory: flat (N, dim) vars — NOT split into heads
        mem_vals, mem_vars = memory.read(torch.arange(N, device=device))
        # mem_vals: (N, dim), mem_vars: (N, dim)
        mem_vals_split = self.split_q(mem_vals)                      # (N, heads, head_dim)

        # Evolution bank: (N, W, dim) and (N, W)
        evo_vals, evo_times = evo_bank.read(torch.arange(N, device=device))

        edge_emb_kv   = self.split_kv(self.edge_proj(edge_emb))
        edge_emb_full = self.expand_kv(edge_emb_kv)                 # (E, heads, head_dim)

        # Recency bias per EDGE (not per node): index edge src into time_emb
        # time_emb is (E, dim); project to (E, heads)
        recency_bias = self.recency_bias_proj(time_emb)              # (E, heads) ← FIXED

        out = self.propagate(
            edge_index,
            Q=Q, K=K, V=V,
            mem_val=mem_vals_split,
            mem_var=mem_vars,        # flat (N, dim) — will arrive as (E, dim) after gather
            hidden=mem_vals,         # (N, dim) — same gather
            evo_vals=evo_vals,       # (N, W, dim)
            evo_times=evo_times,     # (N, W)
            edge_emb=edge_emb_full,
            node_temp=temp,          # (N, heads)
            recency_bias=recency_bias,  # (E, heads) — passed directly, not gathered
            node_last_time=node_last_time,  # (N,)
            size=(N, N)
        )

        agg = out.view(N, self.dim)

        # Recurrent fusion with GRU hidden state
        h   = memory.get_hidden(torch.arange(N, device=device))     # (N, dim)
        agg = self.recurrent_fusion(torch.cat([agg, h], dim=-1))

        agg = self.moe(agg)
        agg = self.o_proj(agg)

        # AdaLN conditioned on per-node mean time embedding
        # Scatter mean of time_emb over src nodes
        src       = edge_index[0]
        time_cond = torch.zeros(N, self.dim, device=device)
        count_cond = torch.zeros(N, 1, device=device)
        time_cond.scatter_add_(0, src.unsqueeze(1).expand(-1, self.dim), time_emb)
        count_cond.scatter_add_(0, src.unsqueeze(1), torch.ones(E_num, 1, device=device))
        time_cond = time_cond / (count_cond + 1e-6)                  # (N, dim)

        agg  = self.ada_norm(agg, time_cond)
        beta = torch.sigmoid(self.gate_lin(x))
        return beta * x + (1 - beta) * agg

    def message(self, Q_i, K_j, V_j,
                mem_val_i, mem_val_j,
                mem_var_i, mem_var_j,     # (E, dim) flat ← FIXED
                hidden_i, hidden_j,        # (E, dim)
                evo_vals_j, evo_times_j,  # (E, W, dim), (E, W)
                edge_emb,
                node_temp_i,              # (E, heads)
                recency_bias,             # (E, heads) ← passed through, no gather needed
                node_last_time_i,         # (E,)
                index):

        K_j  = K_j + edge_emb
        attn = (Q_i * K_j).sum(dim=-1) / (self.head_dim ** 0.5)     # (E, heads)
        attn = attn / node_temp_i.clamp(min=0.01)
        attn = attn + recency_bias                                    # (E, heads)

        # Trust from evolution — vars are already (E, dim) flat
        trust = self.trust_net(
            hidden_i, hidden_j,
            evo_vals_j, evo_times_j,
            mem_var_i, mem_var_j,        # (E, dim) ← no shape mismatch now
            node_last_time_i
        )                                                             # (E,)

        self.trust_cache = trust.detach()

        attn = attn * trust.unsqueeze(-1)                            # broadcast over heads
        attn = softmax(attn, index)
        attn = self.dropout(attn)
        return V_j * attn.unsqueeze(-1)                              # (E, heads, head_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Full Model
# ═══════════════════════════════════════════════════════════════════════════════

class RecurrentEvolutionGNN(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_dim,
        edge_dim,
        hidden_dim,
        num_layers      = 2,
        heads           = 8,
        kv_heads        = 2,
        memory_momentum = 0.9,
        num_experts     = 4,
        moe_top_k       = 2,
        dropout         = 0.1,
        evo_window      = 8,
        freq_decay      = 0.95,
    ):
        super().__init__()
        self.num_nodes  = num_nodes
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_enc   = nn.Linear(edge_dim, hidden_dim)
        self.time_enc   = FourierTimeEncoding(hidden_dim)

        self.memory      = RecurrentNodeMemory(num_nodes, hidden_dim, momentum=memory_momentum)
        self.freq_tracker = FrequencyTracker(num_nodes, decay=freq_decay)
        self.evo_bank    = NeighborhoodEvolutionBank(num_nodes, hidden_dim, window=evo_window)

        self.layers = nn.ModuleList([
            RecurrentEvolutionAttention(
                hidden_dim, heads=heads, kv_heads=kv_heads,
                dropout=dropout, num_experts=num_experts,
                moe_top_k=moe_top_k, evo_window=evo_window
            ) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch):
        N          = batch.x.size(0)
        device     = batch.x.device
        edge_index = batch.edge_index
        src        = edge_index[0]
        E_num      = src.size(0)

        x        = self.input_proj(batch.x)
        edge_emb = self.edge_enc(batch.msg)
        time_emb = self.time_enc(batch.t)                            # (E, hidden)

        # Per-node last-seen time via scatter max
        node_last_time = torch.zeros(N, device=device)
        node_last_time.scatter_reduce_(
            0, src, batch.t.float(), reduce='amax', include_self=True)

        # Burstiness: (N, 1)
        node_burst = self.freq_tracker.read(
            torch.arange(N, device=device)).unsqueeze(-1).float()

        for layer in self.layers:
            x = layer(
                x, edge_index, edge_emb, time_emb,
                node_last_time, node_burst,
                self.memory, self.evo_bank
            )

        x = self.final_norm(x)

        # ── Updates ─────────────────────────────────────────────────────
        all_idx = torch.arange(N, device=device)
        self.memory.write(all_idx, x)

        active_src = src.unique()
        self.freq_tracker.update(active_src, node_last_time[active_src])

        # Neighborhood mean for evolution bank
        dst          = edge_index[1]
        neigh_mean   = torch.zeros(N, self.hidden_dim, device=device)
        neigh_count  = torch.zeros(N, 1, device=device)
        neigh_mean.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.hidden_dim), x[src])
        neigh_count.scatter_add_(0, dst.unsqueeze(1), torch.ones(E_num, 1, device=device))
        neigh_mean   = neigh_mean / (neigh_count + 1e-6)

        active_dst = dst.unique()
        self.evo_bank.write(active_dst, neigh_mean[active_dst], node_last_time[active_dst])

        return x





# ═══════════════════════════════════════════════════════════════════════════════
# Sanity check
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from types import SimpleNamespace

    N, E_num = 200, 600
    IN_DIM, EDGE_DIM, HIDDEN = 1, 172, 64

    batch = SimpleNamespace(
        x          = torch.randn(N, IN_DIM),
        msg        = torch.randn(E_num, EDGE_DIM),
        t          = torch.rand(E_num) * 1000,
        edge_index = torch.randint(0, N, (2, E_num))
    )

    model = RecurrentEvolutionGNN(
        num_nodes  = N,
        in_dim     = IN_DIM,
        edge_dim   = EDGE_DIM,
        hidden_dim = HIDDEN,
        num_layers = 2,
        heads      = 8,
        kv_heads   = 2,
        evo_window = 8,
    )

    out = model(batch)
    print(f"Output shape : {out.shape}")        # (200, 64) ✓
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,}")
    print("Sanity check passed ✓")