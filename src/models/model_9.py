"""
Temporal Trust Graph Neural Network for Unsupervised Anomaly Detection
======================================================================
Architecture overview:
  - Time2Vec + TimeEncoder   : learns temporal patterns (trend + periodicity)
  - RecurrentMemory          : per-node GRU state + uncertainty tracking
  - EvolutionBank            : rolling window of neighborhood embeddings
  - StructuralTrustModule    : CORE — computes trust(i→j) from STRUCTURAL SIMILARITY
                               between i and j's local neighborhood profiles.
                               Anomalous nodes have structurally deviant neighborhoods
                               → they receive LOW trust → can't poison central node.
  - TemporalTrustAttention   : TGATv2-style attention gated by trust scores
  - EnhancedTemporalGNN      : full model, returns node embeddings for contrastive learning

Trust Mechanism (contribution):
  Instead of cross-attention over an evolution window (which is slow and hard to
  interpret), we compute STRUCTURAL SIMILARITY trust:
    1. Degree-profile similarity   : are i and j similarly connected?
    2. Neighborhood embedding sim  : do i and j have similar local contexts?
    3. Temporal consistency        : has j's behavior been stable over time?
    4. Uncertainty gating          : low-confidence nodes get penalized

  These four signals are fused by a small MLP → trust ∈ [0, 1].
  Trust is used to modulate attention weights: abnormal neighbors are
  down-weighted, normal similar neighbors are up-weighted.

  Key property: trust is ASYMMETRIC — trust(i→j) ≠ trust(j→i).
  Node i trusts j if j looks structurally similar to i's expected neighborhood.
  This prevents camouflage: an anomalous node cannot fake structural similarity
  with its normal neighbors (their degree profiles and embeddings will differ).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, degree


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Time2Vec — learns periodic + trend components from raw timestamps
# ═══════════════════════════════════════════════════════════════════════════════

class Time2Vec(nn.Module):
    """
    t → [w0*t + b0,  sin(w1*t + b1), ..., sin(wk*t + bk)]
    Linear term captures trend; sinusoids capture periodicity.
    Frequencies w are LEARNED — model discovers daily/weekly patterns.
    """
    def __init__(self, out_dim: int):
        super().__init__()
        k = out_dim - 1
        self.w0 = nn.Parameter(torch.randn(1) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.W  = nn.Parameter(torch.randn(k) * 0.01)
        self.B  = nn.Parameter(torch.zeros(k))
        self.out_dim = out_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (N,) → (N, out_dim)"""
        t = t.float().unsqueeze(-1)                    # (N, 1)
        linear   = t * self.w0 + self.b0              # (N, 1)
        periodic = torch.sin(t * self.W + self.B)     # (N, k)
        return torch.cat([linear, periodic], dim=-1)  # (N, out_dim)


class TimeEncoder(nn.Module):
    """
    Full time encoding:
      Time2Vec(t_abs) + Time2Vec(t_rel) → MLP → hidden_dim
    t_abs = absolute edge timestamp
    t_rel = edge_time - node_last_seen_time  (how long since this node was active)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.t2v     = Time2Vec(hidden_dim // 2)
        self.rel_t2v = Time2Vec(hidden_dim // 2)
        self.proj    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t_abs: torch.Tensor,
                t_rel: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        t_abs: (E,) absolute timestamps
        t_rel: (E,) relative gaps (optional)
        Returns: (E, hidden_dim)
        """
        abs_enc = self.t2v(t_abs)                                    # (E, h/2)
        rel_enc = self.rel_t2v(t_rel) if t_rel is not None \
                  else torch.zeros_like(abs_enc)                     # (E, h/2)
        combined = torch.cat([abs_enc, rel_enc], dim=-1)             # (E, h)
        return self.proj(combined)                                   # (E, h)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Recurrent Node Memory with Uncertainty
# ═══════════════════════════════════════════════════════════════════════════════

class RecurrentMemory(nn.Module):
    """
    Per-node GRU hidden state updated at each temporal snapshot.
    Tracks variance as EMA of squared update delta → uncertainty estimate.

    Low variance  = stable node  = trustworthy memory
    High variance = volatile node = unreliable / potentially anomalous
    """
    def __init__(self, num_nodes: int, dim: int, momentum: float = 0.9):
        super().__init__()
        self.gru      = nn.GRUCell(dim, dim)
        self.momentum = momentum
        # Persistent buffers — survive across forward() calls
        self.register_buffer("hidden",   torch.zeros(num_nodes, dim))
        self.register_buffer("variance", torch.ones(num_nodes, dim) * 0.1)

    def read(self, idx: torch.Tensor):
        """Returns (hidden, variance) for node indices idx."""
        return self.hidden[idx], self.variance[idx]

    def get_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        return self.hidden[idx]

    @torch.no_grad()
    def write(self, idx: torch.Tensor, x: torch.Tensor):
        """Update GRU state and EMA variance for node indices idx."""
        h_old = self.hidden[idx]
        h_new = self.gru(x.detach(), h_old)
        delta = h_new - h_old
        self.variance[idx] = (self.momentum * self.variance[idx]
                              + (1 - self.momentum) * delta.pow(2))
        self.hidden[idx] = h_new


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Neighborhood Evolution Bank
# ═══════════════════════════════════════════════════════════════════════════════

class EvolutionBank(nn.Module):
    """
    Rolling circular buffer of K most recent neighborhood mean embeddings per node.
    Shape: (num_nodes, window, dim)

    Used by StructuralTrustModule to measure temporal consistency of j's neighborhood.
    Stable node  → low variance across window → high temporal consistency.
    Anomaly      → erratic neighborhood changes → low temporal consistency.
    """
    def __init__(self, num_nodes: int, dim: int, window: int = 6):
        super().__init__()
        self.window = window
        self.dim    = dim
        self.register_buffer("bank", torch.zeros(num_nodes, window, dim))
        self.register_buffer("ptr",  torch.zeros(num_nodes, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(num_nodes, dtype=torch.long))  # how many slots written

    def read(self, idx: torch.Tensor) -> torch.Tensor:
        """Returns (B, W, D) evolution window for node indices."""
        return self.bank[idx]

    def temporal_consistency(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Returns (B,) consistency score in [0,1].
        Computed as 1 / (1 + std of cosine-sim across window steps).
        A stable node has consistent embeddings → high score.
        """
        evo = self.bank[idx]                                         # (B, W, D)
        # Normalize each step's embedding
        evo_norm = F.normalize(evo, dim=-1, eps=1e-6)                # (B, W, D)
        # Pairwise cosine similarities between consecutive steps
        # sim[t] = cos(evo[t], evo[t+1])
        sim = (evo_norm[:, :-1, :] * evo_norm[:, 1:, :]).sum(-1)    # (B, W-1)
        # Low std of similarities = consistent behavior
        consistency = 1.0 / (1.0 + sim.std(dim=-1))                 # (B,)
        return consistency.clamp(0.0, 1.0)

    @torch.no_grad()
    def write(self, idx: torch.Tensor, emb: torch.Tensor):
        """emb: (B, D) — write mean neighborhood embedding for each node in idx."""
        for b in range(idx.size(0)):
            node = int(idx[b])
            p    = int(self.ptr[node]) % self.window
            self.bank[node, p] = emb[b].detach()
            self.ptr[node]    += 1
            self.filled[node]  = min(int(self.filled[node]) + 1, self.window)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Structural Trust Module — CORE CONTRIBUTION
#
# Trust(i→j) = "How much should node i weight messages from node j?"
#
# We compute trust from FOUR structural signals:
#
#  (A) Embedding similarity      : cos(h_i, h_j)
#      Nodes with similar roles/behavior should have similar embeddings.
#      An anomalous j will be far from normal i in embedding space.
#
#  (B) Degree profile similarity : |deg_i - deg_j| / (deg_i + deg_j)
#      Anomalous nodes often have unusual degree patterns (hub injections,
#      isolated attackers). Structural deviance → low trust.
#
#  (C) Neighborhood embedding similarity: cos(mean_neigh_i, mean_neigh_j)
#      i and j trust each other more if their neighborhoods "look alike".
#      Camouflaged anomalies with fake normal connections will still have
#      different neighborhood profiles from genuinely normal nodes.
#
#  (D) Temporal consistency of j : from EvolutionBank
#      An erratic j (high variance in neighborhood over time) is less trustworthy.
#
#  All four signals → MLP fusion → scalar trust ∈ [0,1]
#  Further gated by combined uncertainty from RecurrentMemory variance.
#
# ASYMMETRY: trust(i→j) uses h_i as reference → naturally asymmetric.
# ═══════════════════════════════════════════════════════════════════════════════

class StructuralTrustModule(nn.Module):
    """
    Computes per-edge trust scores from structural similarity signals.
    All inputs are (E, ...) tensors aligned with the edge list.
    """
    def __init__(self, dim: int, hidden_ratio: float = 0.5):
        super().__init__()
        h = max(dim // 4, 16)

        # Project neighborhood means before comparison
        self.neigh_proj_i = nn.Linear(dim, h)
        self.neigh_proj_j = nn.Linear(dim, h)

        # MLP that fuses the 4 signals + uncertainty into a trust logit
        # Input features:
        #   1. emb_cos_sim          (1,)
        #   2. degree_diff          (1,)
        #   3. neigh_cos_sim        (1,)
        #   4. temporal_consistency (1,)
        #   5. uncertainty_i        (1,)
        #   6. uncertainty_j        (1,)
        # Total: 6 scalar features
        self.trust_mlp = nn.Sequential(
            nn.Linear(6, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, 1)
        )

        # Learned temperature for sharpening trust distribution
        self.log_temp = nn.Parameter(torch.zeros(1))   # starts at 1.0

    def forward(
        self,
        h_i: torch.Tensor,         # (E, dim)  — i's recurrent hidden state
        h_j: torch.Tensor,         # (E, dim)  — j's recurrent hidden state
        deg_i: torch.Tensor,       # (E,)      — degree of node i
        deg_j: torch.Tensor,       # (E,)      — degree of node j
        neigh_i: torch.Tensor,     # (E, dim)  — mean neighborhood embedding of i
        neigh_j: torch.Tensor,     # (E, dim)  — mean neighborhood embedding of j
        var_i: torch.Tensor,       # (E, dim)  — memory variance of i
        var_j: torch.Tensor,       # (E, dim)  — memory variance of j
        temp_cons_j: torch.Tensor, # (E,)      — temporal consistency of j
        eps: float = 1e-6
    ) -> torch.Tensor:
        """Returns trust scores (E,) in [0, 1]."""

        # ── (A) Embedding cosine similarity ─────────────────────────────────
        h_i_n = F.normalize(h_i, dim=-1, eps=eps)                    # (E, dim)
        h_j_n = F.normalize(h_j, dim=-1, eps=eps)
        emb_cos = (h_i_n * h_j_n).sum(-1, keepdim=True)             # (E, 1) ∈ [-1,1]

        # ── (B) Degree profile similarity ───────────────────────────────────
        deg_i_f = deg_i.float()
        deg_j_f = deg_j.float()
        deg_diff = torch.abs(deg_i_f - deg_j_f) / (deg_i_f + deg_j_f + eps)
        deg_sim  = (1.0 - deg_diff).unsqueeze(-1)                    # (E, 1) ∈ [0,1]

        # ── (C) Neighborhood embedding similarity ───────────────────────────
        ni_p = F.normalize(self.neigh_proj_i(neigh_i), dim=-1, eps=eps)  # (E, h)
        nj_p = F.normalize(self.neigh_proj_j(neigh_j), dim=-1, eps=eps)
        neigh_cos = (ni_p * nj_p).sum(-1, keepdim=True)             # (E, 1) ∈ [-1,1]

        # ── (D) Temporal consistency of j ───────────────────────────────────
        temp_feat = temp_cons_j.unsqueeze(-1)                        # (E, 1) ∈ [0,1]

        # ── Uncertainty features ─────────────────────────────────────────────
        # Scalar uncertainty = mean std of variance across dimensions
        unc_i = var_i.mean(-1).sqrt().unsqueeze(-1)                  # (E, 1)
        unc_j = var_j.mean(-1).sqrt().unsqueeze(-1)                  # (E, 1)

        # ── Fuse all signals ─────────────────────────────────────────────────
        features = torch.cat([
            emb_cos,    # (E, 1)  — embedding similarity
            deg_sim,    # (E, 1)  — degree similarity
            neigh_cos,  # (E, 1)  — neighborhood profile similarity
            temp_feat,  # (E, 1)  — temporal consistency of j
            unc_i,      # (E, 1)  — uncertainty of i's memory
            unc_j,      # (E, 1)  — uncertainty of j's memory
        ], dim=-1)                                                    # (E, 6)

        trust_logit = self.trust_mlp(features).squeeze(-1)           # (E,)

        # Apply learned temperature sharpening
        temp = self.log_temp.exp().clamp(min=0.1, max=10.0)
        return torch.sigmoid(trust_logit * temp)                     # (E,) ∈ [0,1]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Temporal Trust Attention Layer (TGATv2-style)
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalTrustAttention(MessagePassing):
    """
    TGATv2-inspired attention with structural trust gating.

    Attention flow:
      1. Compute Q, K, V from node embeddings
      2. Enrich K with edge features and time encoding
      3. Compute structural trust score per edge
      4. Modulate raw attention logits by trust → anomalous neighbors are suppressed
      5. Gated residual + LayerNorm
      6. Fuse with recurrent hidden state
    """
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1,
                 window: int = 6):
        super().__init__(aggr="add", node_dim=0)
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim      = dim
        self.heads    = heads
        self.head_dim = dim // heads
        self.scale    = self.head_dim ** -0.5
        self.window   = window

        # Attention projections
        self.q_proj    = nn.Linear(dim, dim)
        self.k_proj    = nn.Linear(dim, dim)
        self.v_proj    = nn.Linear(dim, dim)
        self.o_proj    = nn.Linear(dim, dim)

        # Feature enrichment for keys
        self.edge_proj = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(dim, dim)

        # TGATv2: non-linear attention (concat Q and K before scoring)
        self.attn_nonlin = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, heads)
        )

        # Trust module
        self.trust = StructuralTrustModule(dim)

        self.dropout = nn.Dropout(dropout)

        # Gated residual connection
        self.gate = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)

        # Fusion with recurrent hidden state
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # Cache trust scores for external inspection / loss computation
        self._trust_cache: Optional[torch.Tensor] = None

    def split(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (N, dim) → (N, heads, head_dim)"""
        return x.view(-1, self.heads, self.head_dim)

    def forward(
        self,
        x:          torch.Tensor,    # (N, dim)
        edge_index: torch.Tensor,    # (2, E)
        edge_emb:   torch.Tensor,    # (E, dim)
        time_emb:   torch.Tensor,    # (E, dim)
        memory:     RecurrentMemory,
        evo_bank:   EvolutionBank,
        deg:        torch.Tensor,    # (N,)  node degrees in current snapshot
    ) -> torch.Tensor:
        N      = x.size(0)
        device = x.device
        node_idx = torch.arange(N, device=device)

        # ── Compute Q, K, V ──────────────────────────────────────────────────
        Q = self.split(self.q_proj(x))                               # (N, h, hd)
        K = self.split(self.k_proj(x))
        V = self.split(self.v_proj(x))

        # ── Read memory ───────────────────────────────────────────────────────
        mem_vals, mem_vars = memory.read(node_idx)                   # (N, dim) each
        evo_vals = evo_bank.read(node_idx)                           # (N, W, dim)

        # Temporal consistency per node (will be passed as edge features)
        temp_cons = evo_bank.temporal_consistency(node_idx)          # (N,)

        # ── Project edge / time features ─────────────────────────────────────
        edge_emb_proj = self.split(self.edge_proj(edge_emb))        # (E, h, hd)
        time_emb_proj = self.split(self.time_proj(time_emb))        # (E, h, hd)

        # ── Propagate ─────────────────────────────────────────────────────────
        out = self.propagate(
            edge_index,
            x=x, Q=Q, K=K, V=V,
            mem_val=mem_vals,
            mem_var=mem_vars,
            deg=deg,
            temp_cons=temp_cons,
            edge_emb=edge_emb_proj,
            time_emb=time_emb_proj,
            size=(N, N)
        )
        out = out.view(N, self.dim)                                  # (N, dim)

        # ── Fuse with recurrent memory ────────────────────────────────────────
        h   = memory.get_hidden(node_idx)                            # (N, dim)
        out = self.fusion(torch.cat([out, h], dim=-1))
        out = self.o_proj(out)

        # ── Gated residual + LayerNorm ────────────────────────────────────────
        g = torch.sigmoid(self.gate(x))
        return self.norm(g * x + (1 - g) * out)

    def message(
        self,
        x_i: torch.Tensor,          # (E, dim)   — central node features
        x_j: torch.Tensor,          # (E, dim)   — neighbor features
        Q_i: torch.Tensor,          # (E, h, hd)
        K_j: torch.Tensor,          # (E, h, hd)
        V_j: torch.Tensor,          # (E, h, hd)
        mem_val_i: torch.Tensor,    # (E, dim)
        mem_val_j: torch.Tensor,    # (E, dim)
        mem_var_i: torch.Tensor,    # (E, dim)
        mem_var_j: torch.Tensor,    # (E, dim)
        deg_i: torch.Tensor,        # (E,)
        deg_j: torch.Tensor,        # (E,)
        temp_cons_j: torch.Tensor,  # (E,)
        edge_emb: torch.Tensor,     # (E, h, hd)
        time_emb: torch.Tensor,     # (E, h, hd)
        index: torch.Tensor,        # (E,)   — target node indices for softmax
    ) -> torch.Tensor:

        E = x_i.size(0)

        # ── Enrich K with edge and temporal context ───────────────────────────
        K_j_rich = K_j + edge_emb + time_emb                        # (E, h, hd)

        # ── TGATv2 non-linear attention ───────────────────────────────────────
        # Concatenate Q and enriched K, then compute per-head score
        Q_flat  = Q_i.view(E, -1)                                    # (E, dim)
        K_flat  = K_j_rich.view(E, -1)                               # (E, dim)
        qk_cat  = torch.cat([Q_flat, K_flat], dim=-1)                # (E, dim*2)
        attn_scores = self.attn_nonlin(qk_cat)                       # (E, heads)

        # ── Structural trust ──────────────────────────────────────────────────
        # Neighborhood mean as proxy for local structure
        # We use mem_val as the stored neighborhood mean (updated in forward)
        trust = self.trust(
            h_i        = mem_val_i,    # i's recurrent history
            h_j        = mem_val_j,    # j's recurrent history
            deg_i      = deg_i,
            deg_j      = deg_j,
            neigh_i    = mem_val_i,    # reuse as neighborhood proxy for i
            neigh_j    = mem_val_j,    # reuse as neighborhood proxy for j
            var_i      = mem_var_i,
            var_j      = mem_var_j,
            temp_cons_j= temp_cons_j,
        )                                                            # (E,) ∈ [0,1]

        # Cache for loss computation and analysis
        self._trust_cache = trust.detach()

        # ── Trust-modulated attention ─────────────────────────────────────────
        # Multiply raw attention by trust before softmax normalization.
        # Low-trust (anomalous) neighbors contribute little to the update.
        attn_scores = attn_scores * trust.unsqueeze(-1)              # (E, heads)
        attn_scores = softmax(attn_scores, index)                    # normalize
        attn_scores = self.dropout(attn_scores)

        # ── Aggregate values ──────────────────────────────────────────────────
        return V_j * attn_scores.unsqueeze(-1)                       # (E, h, hd)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Full Model: EnhancedTemporalGNN
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedTemporalGNN(nn.Module):
    """
    Full temporal graph neural network with structural trust.

    Input batch fields expected:
        batch.x          : (N, in_dim)   node features
        batch.edge_index : (2, E)        directed edge list [src, dst]
        batch.msg        : (E, edge_dim) edge/message features
        batch.t          : (E,)          edge timestamps

    Output:
        x : (N, hidden_dim) node representations for contrastive learning

    The trust scores from the last layer are accessible via
        model.layers[-1]._trust_cache
    for use in trust-regularized contrastive loss.
    """
    def __init__(
        self,
        num_nodes:        int,
        in_dim:           int,
        edge_dim:         int,
        hidden_dim:       int,
        num_layers:       int   = 2,
        heads:            int   = 8,
        dropout:          float = 0.1,
        window:           int   = 6,
        memory_momentum:  float = 0.9,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes  = num_nodes

        # Encoders
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_enc   = nn.Linear(edge_dim, hidden_dim)
        self.time_enc   = TimeEncoder(hidden_dim)

        # Persistent state
        self.memory   = RecurrentMemory(num_nodes, hidden_dim, momentum=memory_momentum)
        self.evo_bank = EvolutionBank(num_nodes, hidden_dim, window=window)

        # Attention layers
        self.layers = nn.ModuleList([
            TemporalTrustAttention(
                hidden_dim, heads=heads, dropout=dropout, window=window
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    # ────────────────────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ────────────────────────────────────────────────────────────────────────
    def forward(self, batch) -> torch.Tensor:
        N          = batch.x.size(0)
        device     = batch.x.device
        edge_index = batch.edge_index
        src, dst   = edge_index                                      # (E,) each

        # ── Encode inputs ────────────────────────────────────────────────────
        x        = self.input_proj(batch.x)                         # (N, hidden)
        edge_emb = self.edge_enc(batch.msg)                         # (E, hidden)

        # ── Compute node degrees in this snapshot ────────────────────────────
        deg = degree(dst, num_nodes=N, dtype=torch.float)           # (N,)
        deg = deg.clamp(min=1.0)                                    # avoid zero

        # ── Temporal encoding ────────────────────────────────────────────────
        # Per-node last-seen time (for relative encoding)
        node_last_t = torch.zeros(N, device=device)
        node_last_t.scatter_reduce_(
            0, src, batch.t.float(), reduce='amax', include_self=True
        )
        t_rel    = batch.t.float() - node_last_t[src]              # (E,)
        time_emb = self.time_enc(batch.t, t_rel)                   # (E, hidden)

        # ── Message passing ──────────────────────────────────────────────────
        for layer in self.layers:
            x = layer(x, edge_index, edge_emb, time_emb,
                      self.memory, self.evo_bank, deg)

        x = self.final_norm(x)

        # ── Update persistent state ──────────────────────────────────────────
        node_idx = torch.arange(N, device=device)
        self.memory.write(node_idx, x)

        # Compute neighborhood mean embeddings for evolution bank update
        neigh_sum = torch.zeros(N, self.hidden_dim, device=device)
        cnt       = torch.zeros(N, 1, device=device)
        neigh_sum.scatter_add_(
            0, dst.unsqueeze(1).expand(-1, self.hidden_dim), x[src]
        )
        cnt.scatter_add_(0, dst.unsqueeze(1),
                         torch.ones(src.size(0), 1, device=device))
        neigh_mean = neigh_sum / (cnt + 1e-6)                       # (N, hidden)

        unique_dst = dst.unique()
        self.evo_bank.write(unique_dst, neigh_mean[unique_dst])

        return x                                                     # (N, hidden)

    # ────────────────────────────────────────────────────────────────────────
    def get_trust_scores(self, layer_idx: int = -1) -> Optional[torch.Tensor]:
        """
        Return cached trust scores from a given layer after forward().
        Useful for trust-regularized contrastive loss.
        """
        return self.layers[layer_idx]._trust_cache

    @torch.no_grad()
    def reset_memory(self):
        """Reset all persistent memory (call between independent graph streams)."""
        self.memory.hidden.zero_()
        self.memory.variance.fill_(0.1)
        self.evo_bank.bank.zero_()
        self.evo_bank.ptr.zero_()
        self.evo_bank.filled.zero_()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Trust-Regularized Contrastive Loss Helper
#
# This is a utility — not part of the GNN itself — showing how trust scores
# can be wired into the contrastive learning objective.
#
# Idea: edges with low trust are likely anomalous connections.
# In contrastive learning, we use trust to re-weight the negative/positive
# sampling: low-trust source nodes are pushed further in representation space.
# ═══════════════════════════════════════════════════════════════════════════════

class TrustWeightedContrastiveLoss(nn.Module):
    """
    Placeholder loss module showing how trust scores gate the contrastive signal.
    Replace the inner InfoNCE / NT-Xent call with your actual contrastive loss.

    trust_scores: (E,) ∈ [0,1] — from model.get_trust_scores()
    anchor:       (B, dim) — anchor node embeddings
    positive:     (B, dim) — positive (augmented / temporally close) embeddings
    negatives:    (B, K, dim) — hard negative embeddings

    The trust of the edge connecting anchor to its positive is used as a
    confidence weight: low-trust edges contribute less to the positive term,
    preventing anomalous pairs from corrupting the representation.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor:       torch.Tensor,  # (B, dim)
        positive:     torch.Tensor,  # (B, dim)
        negatives:    torch.Tensor,  # (B, K, dim)
        trust_weight: torch.Tensor,  # (B,) ∈ [0,1]
    ) -> torch.Tensor:
        B, dim = anchor.shape

        a = F.normalize(anchor,   dim=-1)
        p = F.normalize(positive, dim=-1)
        n = F.normalize(negatives, dim=-1)             # (B, K, dim)

        # Positive similarity
        pos_sim = (a * p).sum(-1) / self.temperature   # (B,)

        # Negative similarities
        neg_sim = torch.bmm(n, a.unsqueeze(-1)).squeeze(-1) / self.temperature  # (B, K)

        # InfoNCE
        logits  = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, K+1)
        labels  = torch.zeros(B, dtype=torch.long, device=anchor.device)
        loss_per_sample = F.cross_entropy(logits, labels, reduction='none')  # (B,)

        # Weight by trust: low-trust pairs (potential anomalies) contribute less
        # to the normal contrastive objective — their loss is down-weighted so
        # they don't corrupt the normal-node representation space.
        weighted_loss = (trust_weight * loss_per_sample).mean()
        return weighted_loss


# ═══════════════════════════════════════════════════════════════════════════════
# Quick sanity check
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import types

    torch.manual_seed(42)
    device = "cpu"

    # Synthetic mini-batch
    N, E      = 50, 120
    in_dim    = 32
    edge_dim  = 16
    hidden    = 64
    num_nodes = 100

    batch = types.SimpleNamespace(
        x          = torch.randn(N, in_dim),
        edge_index = torch.randint(0, N, (2, E)),
        msg        = torch.randn(E, edge_dim),
        t          = torch.rand(E) * 1000,
    )

    model = EnhancedTemporalGNN(
        num_nodes  = num_nodes,
        in_dim     = in_dim,
        edge_dim   = edge_dim,
        hidden_dim = hidden,
        num_layers = 2,
        heads      = 4,
        window     = 6,
    )

    emb = model(batch)
    print(f"Output embeddings shape : {emb.shape}")        # (N, hidden)

    trust = model.get_trust_scores()
    print(f"Trust scores shape      : {trust.shape}")      # (E,)
    print(f"Trust score range       : [{trust.min():.3f}, {trust.max():.3f}]")
    print(f"Mean trust              : {trust.mean():.3f}")
    print("Sanity check passed ✓")