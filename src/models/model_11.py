"""
Temporal Trust GNN for Unsupervised Anomaly Detection
=====================================================
Produces node representations for downstream contrastive learning.
No labels required — fully unsupervised.

Trust Mechanism  trust(i->j) in (0, 2):
  For every edge (i, j) in the current snapshot:

  1. K-VIEW SNAPSHOTS  [SnapshotGNN — TransformerConv]
     The EvolutionBank stores the last K activity snapshots per node.
     Each snapshot is produced by a 2-layer TransformerConv GNN that
     performs transformer-style multi-head attention between a node and
     its one-hop neighbours, incorporating edge_attr directly:
         alpha(i,j) = softmax( (W_Q*x_i)^T (W_K*x_j + W_E*edge_attr) )
         out_i      = W_O * sum_j [ alpha * (W_V*x_j + W_E*edge_attr) ]
     Time is NOT used here — kept clean for the positional encoding step.

  2. TEMPORAL POSITIONAL ENCODING  [RoPE + Time2Vec]
     Before comparing K views, each snapshot embedding is enriched with
     its timestamp using two complementary encodings:

     (a) RoPE (Rotary Position Encoding) for continuous timestamps:
         Pairs of dimensions are rotated by theta = t / T_d where T_d
         are learned per-dimension timescales.
         Key property: dot product between two RoPE-encoded views naturally
         reflects their temporal distance — close-in-time views stay similar,
         distant-in-time views diverge in embedding space.

     (b) Time2Vec (additive):
         Absolute timestamp encoded as [linear + sinusoids] and added.
         Captures periodic patterns (daily/weekly activity cycles).

     Combined: view_k = RoPE(emb_k, t_k) + Linear(Time2Vec(t_k))

  3. K-VIEW TRAJECTORY SIMILARITY
     Position-aligned cosine similarity averaged over K:
         traj_sim = mean_k  cos(view_i_k, view_j_k)

  4. SIGNAL FUSION
     trust_logit = MLP([emb_cos, deg_sim, traj_sim])   (3 scalars)
     trust       = 1 + tanh(trust_logit)               in (0, 2)

     trust > 1 : similar neighbor  -> amplify its contribution
     trust < 1 : deviant neighbor  -> suppress its contribution
     trust -> 0: strongly anomalous -> near-zero influence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.utils import softmax, degree


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Time2Vec
# ═══════════════════════════════════════════════════════════════════════════════

class Time2Vec(nn.Module):
    """
    Encodes a scalar timestamp into a vector:
        [w0*t + b0,  sin(w1*t + b1), ..., sin(w_{k}*t + b_{k})]
    Linear term = trend.  Sinusoids = periodicity.  All frequencies learned.
    """
    def __init__(self, out_dim: int):
        super().__init__()
        assert out_dim >= 2
        k = out_dim - 1
        self.w0 = nn.Parameter(torch.randn(1) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.W  = nn.Parameter(torch.randn(k) * 0.01)
        self.B  = nn.Parameter(torch.zeros(k))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (*,) -> (*, out_dim)"""
        t   = t.float().unsqueeze(-1)
        lin = t * self.w0 + self.b0
        per = torch.sin(t * self.W + self.B)
        return torch.cat([lin, per], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  TimeEncoder  (edge-level: absolute + relative timestamp -> hidden_dim)
# ═══════════════════════════════════════════════════════════════════════════════

class TimeEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        half = hidden_dim // 2
        self.t2v_abs = Time2Vec(half)
        self.t2v_rel = Time2Vec(half)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t_abs: torch.Tensor,
                t_rel: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(E,), (E,) -> (E, hidden_dim)"""
        a = self.t2v_abs(t_abs)
        r = self.t2v_rel(t_rel) if t_rel is not None else torch.zeros_like(a)
        return self.proj(torch.cat([a, r], dim=-1))


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  RecurrentMemory  (per-node GRU hidden state)
# ═══════════════════════════════════════════════════════════════════════════════

class RecurrentMemory(nn.Module):
    """
    Maintains one GRU hidden state per node, updated after each snapshot.
    The hidden state is the running summary of the node's full history.
    """
    def __init__(self, num_nodes: int, dim: int, momentum: float = 0.9):
        super().__init__()
        self.gru      = nn.GRUCell(dim, dim)
        self.momentum = momentum
        self.register_buffer("hidden",   torch.zeros(num_nodes, dim))
        self.register_buffer("variance", torch.ones(num_nodes, dim) * 0.1)

    def read(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.hidden[idx], self.variance[idx]

    def get_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        return self.hidden[idx]

    @torch.no_grad()
    def write(self, idx: torch.Tensor, x: torch.Tensor):
        h_old = self.hidden[idx]
        h_new = self.gru(x.detach(), h_old)
        delta = h_new - h_old
        self.variance[idx] = (self.momentum * self.variance[idx]
                              + (1 - self.momentum) * delta.pow(2))
        self.hidden[idx] = h_new


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SnapshotGNN  — TransformerConv-based node encoder
#
#  Produces the snapshot embedding stored in the EvolutionBank.
#  Uses PyG's TransformerConv which performs full transformer-style
#  multi-head attention between a node and its one-hop neighbours,
#  incorporating edge_attr directly into the key/value computation:
#
#      alpha(i,j) = softmax( (W_Q * x_i)^T (W_K * x_j + W_E * edge_attr) )
#      out_i      = W_O * sum_j [ alpha(i,j) * (W_V * x_j + W_E * edge_attr) ]
#
#  Inputs: node features (x) and edge features (edge_attr) ONLY.
#  Time is NOT used here — it is handled separately in the positional
#  encoding step when comparing K-view trajectories in the trust module.
#
#  Two TransformerConv layers with a residual connection give the model
#  capacity to capture both direct neighbours and two-hop structure.
# ═══════════════════════════════════════════════════════════════════════════════

class SnapshotGNN(nn.Module):
    """
    Transformer-based snapshot encoder using PyG's TransformerConv.

    Produces a rich node embedding from node features and edge attributes,
    capturing neighbourhood structure via transformer-style attention.
    Time is deliberately excluded — it is injected separately during
    the K-view positional encoding in StructuralTrustModule.

    Args:
        dim      : hidden dimension (input, output, and internal)
        heads    : number of attention heads for TransformerConv
        dropout  : dropout on attention weights
    """
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"

        # Layer 1: node features + edge_attr -> dim
        self.conv1 = TransformerConv(
            in_channels  = dim,
            out_channels = dim // heads,
            heads        = heads,
            dropout      = dropout,
            edge_dim     = dim,        # edge_attr dimension matches hidden dim
            concat       = True,       # concat heads -> dim
            beta         = True,       # learnable skip connection inside conv
        )
        self.norm1 = nn.LayerNorm(dim)

        # Layer 2: refine with a second transformer hop
        self.conv2 = TransformerConv(
            in_channels  = dim,
            out_channels = dim // heads,
            heads        = heads,
            dropout      = dropout,
            edge_dim     = dim,
            concat       = True,
            beta         = True,
        )
        self.norm2 = nn.LayerNorm(dim)

        # Final projection + residual gate
        self.out_proj = nn.Linear(dim, dim)
        self.gate     = nn.Linear(dim, 1)

    def forward(self,
                x:          torch.Tensor,   # (N, dim)   projected node features
                edge_index: torch.Tensor,   # (2, E)
                edge_attr:  torch.Tensor,   # (E, dim)   projected edge features
                ) -> torch.Tensor:          # (N, dim)
        # Layer 1
        h = self.norm1(F.gelu(self.conv1(x, edge_index, edge_attr)))  # (N, dim)
        # Layer 2 with residual
        h2 = self.norm2(F.gelu(self.conv2(h, edge_index, edge_attr))) # (N, dim)
        h  = h + h2                                                    # residual

        # Gated output projection
        out = self.out_proj(h)
        g   = torch.sigmoid(self.gate(x))
        return g * x + (1 - g) * out                                  # (N, dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  EvolutionBank  — stores last K snapshot embeddings + timestamps per node
#
#  After each forward pass, the snapshot embedding produced by SnapshotGNN
#  is written here alongside its timestamp.
#  Shape: bank (num_nodes, K, dim),  times (num_nodes, K)
# ═══════════════════════════════════════════════════════════════════════════════

class EvolutionBank(nn.Module):
    def __init__(self, num_nodes: int, dim: int, window: int = 8):
        super().__init__()
        self.window = window
        self.dim    = dim
        self.register_buffer("bank",  torch.zeros(num_nodes, window, dim))
        self.register_buffer("times", torch.zeros(num_nodes, window))
        self.register_buffer("ptr",   torch.zeros(num_nodes, dtype=torch.long))

    def read(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (B, K, dim) embeddings and (B, K) timestamps."""
        return self.bank[idx], self.times[idx]

    @torch.no_grad()
    def write(self, idx: torch.Tensor, emb: torch.Tensor, t: torch.Tensor):
        """
        idx : (B,)    node indices
        emb : (B, D)  snapshot embedding from SnapshotGNN
        t   : (B,)    timestamp of this snapshot
        """
        for b in range(idx.size(0)):
            node = int(idx[b])
            p    = int(self.ptr[node]) % self.window
            self.bank[node,  p] = emb[b].detach()
            self.times[node, p] = float(t[b])
            self.ptr[node] += 1

    @torch.no_grad()
    def reset(self):
        self.bank.zero_()
        self.times.zero_()
        self.ptr.zero_()


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TemporalPositionalEncoding  — RoPE + Time2Vec for K-view snapshots
#
#  Applies two complementary positional encodings to each of the K
#  snapshot embeddings before trajectory comparison:
#
#  (a) RoPE (Rotary Position Encoding) adapted for continuous timestamps:
#      Pairs of dimensions in the embedding are rotated by an angle θ = t / T_d
#      where T_d is a learned per-dimension timescale.
#      This injects RELATIVE temporal information: the dot product between
#      two time-encoded views naturally captures their temporal distance.
#      If views i_k and j_k have similar timestamps, their rotated embeddings
#      will be close; if timestamps differ greatly, they will diverge.
#
#  (b) Time2Vec (additive):
#      Encodes the absolute timestamp as [linear, sin(w1*t), ..., sin(wk*t)]
#      and adds a learned projection of it to the embedding.
#      This captures absolute periodicity (daily/weekly patterns).
#
#  Combined: view_k = RoPE(emb_k, t_k) + Linear(Time2Vec(t_k))
#  The two encodings are complementary:
#      RoPE    captures relative temporal distance between views
#      Time2Vec captures absolute temporal position and periodicity
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalPositionalEncoding(nn.Module):
    """
    Applies RoPE + Time2Vec positional encoding to K snapshot embeddings.

    Args:
        dim     : embedding dimension (must be even for RoPE)
        t2v_dim : dimension of Time2Vec encoding before projection
    """
    def __init__(self, dim: int, t2v_dim: int = 16):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for RoPE"
        self.dim     = dim
        self.t2v_dim = t2v_dim

        # RoPE: learned per-dimension timescale (log-space for numerical stability)
        # Shape (dim//2,): one timescale per pair of dimensions
        self.log_timescales = nn.Parameter(
            torch.linspace(0, 9, dim // 2)  # init: timescales from 1 to e^9 ≈ 8000
        )

        # Time2Vec: absolute timestamp encoding
        self.t2v     = Time2Vec(t2v_dim)
        self.t2v_proj = nn.Linear(t2v_dim, dim)

    def _apply_rope(self,
                    embs:  torch.Tensor,   # (B, K, dim)
                    times: torch.Tensor,   # (B, K)
                    ) -> torch.Tensor:     # (B, K, dim)
        """
        Rotate each embedding by an angle proportional to its timestamp.
        For each pair of dimensions (2d, 2d+1):
            theta_d  = t / exp(log_timescales[d])
            [e_{2d}, e_{2d+1}] -> [e_{2d}*cos(theta) - e_{2d+1}*sin(theta),
                                   e_{2d}*sin(theta) + e_{2d+1}*cos(theta)]
        """
        B, K, D = embs.shape
        timescales = self.log_timescales.exp()                       # (D//2,)

        # angles: (B, K, D//2)
        angles = times.unsqueeze(-1) / timescales.unsqueeze(0).unsqueeze(0)

        cos_a = angles.cos()   # (B, K, D//2)
        sin_a = angles.sin()

        # Split embedding into even/odd dimensions
        e_even = embs[..., 0::2]   # (B, K, D//2)
        e_odd  = embs[..., 1::2]

        # Apply rotation
        rot_even = e_even * cos_a - e_odd * sin_a
        rot_odd  = e_even * sin_a + e_odd * cos_a

        # Interleave back: (B, K, D)
        rotated = torch.stack([rot_even, rot_odd], dim=-1)           # (B, K, D//2, 2)
        return rotated.reshape(B, K, D)

    def forward(self,
                embs:  torch.Tensor,   # (B, K, dim)
                times: torch.Tensor,   # (B, K)
                ) -> torch.Tensor:     # (B, K, dim)
        B, K, D = embs.shape

        # (a) RoPE: rotate embedding in-place by timestamp angle
        rope_embs = self._apply_rope(embs, times)                    # (B, K, dim)

        # (b) Time2Vec: additive absolute time signal
        t_flat  = times.reshape(B * K)                               # (B*K,)
        t2v_enc = self.t2v(t_flat).view(B, K, self.t2v_dim)         # (B, K, t2v_dim)
        t2v_add = self.t2v_proj(t2v_enc)                             # (B, K, dim)

        # Combine: rotated embedding + absolute time encoding
        return rope_embs + t2v_add                                   # (B, K, dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  StructuralTrustModule  — CORE CONTRIBUTION
#
#  For every edge (i -> j), computes trust(i, j) in (0, 2).
#
#  [A] EMBEDDING SIMILARITY
#      cos(h_i, h_j) using current GRU hidden states.
#
#  [B] DEGREE PROFILE SIMILARITY
#      1 - |deg_i - deg_j| / (deg_i + deg_j)
#
#  [C] K-VIEW TRAJECTORY SIMILARITY
#      Each node has K snapshot embeddings in EvolutionBank (from TransformerConv).
#      Apply TemporalPositionalEncoding to each view:
#          view_k = RoPE(emb_k, t_k) + Time2Vec_proj(t_k)
#      Then compare position-aligned:
#          traj_sim = mean_k cos(view_i_k, view_j_k)
#
#  FUSION:
#      trust = 1 + tanh( MLP([A, B, C]) )   in (0, 2)
# ═══════════════════════════════════════════════════════════════════════════════

class StructuralTrustModule(nn.Module):
    def __init__(self, dim: int, window: int = 8, t2v_dim: int = 16):
        super().__init__()
        self.window = window

        # Temporal positional encoding: RoPE + Time2Vec
        self.time_pe = TemporalPositionalEncoding(dim, t2v_dim=t2v_dim)

        # Final MLP: 3 scalar signals -> trust logit
        h = max(dim // 4, 16)
        self.trust_mlp = nn.Sequential(
            nn.Linear(3, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

    def forward(
        self,
        h_i:    torch.Tensor,   # (E, dim)    current GRU hidden state of i
        h_j:    torch.Tensor,   # (E, dim)    current GRU hidden state of j
        deg_i:  torch.Tensor,   # (E,)        in-degree of i
        deg_j:  torch.Tensor,   # (E,)        in-degree of j
        embs_i: torch.Tensor,   # (E, K, dim) last K snapshot embeddings of i
        embs_j: torch.Tensor,   # (E, K, dim) last K snapshot embeddings of j
        t_i:    torch.Tensor,   # (E, K)      timestamps of i's K snapshots
        t_j:    torch.Tensor,   # (E, K)      timestamps of j's K snapshots
        eps:    float = 1e-6,
    ) -> torch.Tensor:          # (E,) in (0, 2)

        # ── [A] Embedding similarity ──────────────────────────────────────────
        emb_cos = (F.normalize(h_i, dim=-1, eps=eps)
                   * F.normalize(h_j, dim=-1, eps=eps)
                   ).sum(-1, keepdim=True)                           # (E, 1)

        # ── [B] Degree profile similarity ─────────────────────────────────────
        di, dj  = deg_i.float(), deg_j.float()
        deg_sim = (1.0 - (di - dj).abs() / (di + dj + eps)
                   ).unsqueeze(-1)                                   # (E, 1)

        # ── [C] K-view trajectory similarity ─────────────────────────────────
        # Apply RoPE + Time2Vec to each of the K snapshot embeddings.
        # After encoding, view_k carries both the structural content of the
        # snapshot AND a rotation / additive signal from its timestamp.
        views_i = self.time_pe(embs_i, t_i)                         # (E, K, dim)
        views_j = self.time_pe(embs_j, t_j)

        # Position-aligned cosine similarity, averaged over K
        vi_n     = F.normalize(views_i, dim=-1, eps=eps)
        vj_n     = F.normalize(views_j, dim=-1, eps=eps)
        traj_sim = (vi_n * vj_n).sum(-1).mean(-1, keepdim=True)     # (E, 1)

        # ── Fuse and map to (0, 2) ────────────────────────────────────────────
        features    = torch.cat([emb_cos, deg_sim, traj_sim], dim=-1)  # (E, 3)
        trust_logit = self.trust_mlp(features).squeeze(-1)             # (E,)
        return 1.0 + torch.tanh(trust_logit)                          # (E,) in (0,2)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  TemporalTrustAttention  (one message-passing layer)
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalTrustAttention(MessagePassing):
    """
    TGATv2-style attention gated by structural trust.

    Per-edge (i <- j):
      1. Enrich K_j with edge and time features.
      2. TGATv2 score: LeakyReLU(Linear([Q_i || K_j_rich])) -> per-head scalar.
      3. trust(i->j) from StructuralTrustModule.
      4. scores *= trust  then softmax  -> trust-normalised attention.
      5. Aggregate trust-weighted values -> output.
      6. Gated residual + LayerNorm + fusion with GRU memory.
    """
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1,
                 window: int = 8, t2v_dim: int = 16):
        super().__init__(aggr="add", node_dim=0)
        assert dim % heads == 0
        self.dim      = dim
        self.heads    = heads
        self.head_dim = dim // heads

        self.q_proj    = nn.Linear(dim, dim)
        self.k_proj    = nn.Linear(dim, dim)
        self.v_proj    = nn.Linear(dim, dim)
        self.o_proj    = nn.Linear(dim, dim)
        self.edge_proj = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(dim, dim)

        self.attn_scorer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(dim, heads),
        )

        self.trust   = StructuralTrustModule(dim, window=window, t2v_dim=t2v_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate    = nn.Linear(dim, 1)
        self.norm    = nn.LayerNorm(dim)
        self.fusion  = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self._trust_cache: Optional[torch.Tensor] = None

    # --------------------------------------------------------------------------
    def forward(self,
                x:          torch.Tensor,       # (N, dim)
                edge_index: torch.Tensor,       # (2, E)
                edge_emb:   torch.Tensor,       # (E, dim)
                time_emb:   torch.Tensor,       # (E, dim)
                memory:     RecurrentMemory,
                evo_bank:   EvolutionBank,
                deg:        torch.Tensor,       # (N,)
                ) -> torch.Tensor:

        N    = x.size(0)
        nidx = torch.arange(N, device=x.device)

        Q = self.q_proj(x).view(N, self.heads, self.head_dim)
        K = self.k_proj(x).view(N, self.heads, self.head_dim)
        V = self.v_proj(x).view(N, self.heads, self.head_dim)

        mem_vals, _ = memory.read(nidx)                              # (N, dim)
        embs_all, times_all = evo_bank.read(nidx)                   # (N,K,D),(N,K)

        ep = self.edge_proj(edge_emb).view(-1, self.heads, self.head_dim)
        tp = self.time_proj(time_emb).view(-1, self.heads, self.head_dim)

        out = self.propagate(
            edge_index,
            x=x, Q=Q, K=K, V=V,
            mem_val=mem_vals,
            deg=deg,
            embs=embs_all,
            times=times_all,
            edge_emb=ep,
            time_emb=tp,
            size=(N, N),
        )
        out = out.view(N, self.dim)

        # Fuse with recurrent memory
        h   = memory.get_hidden(nidx)
        out = self.fusion(torch.cat([out, h], dim=-1))
        out = self.o_proj(out)

        g = torch.sigmoid(self.gate(x))
        return self.norm(g * x + (1 - g) * out)

    # --------------------------------------------------------------------------
    def message(self,
                x_i:       torch.Tensor,   # (E, dim)
                x_j:       torch.Tensor,
                Q_i:       torch.Tensor,   # (E, heads, head_dim)
                K_j:       torch.Tensor,
                V_j:       torch.Tensor,
                mem_val_i: torch.Tensor,   # (E, dim)
                mem_val_j: torch.Tensor,
                deg_i:     torch.Tensor,   # (E,)
                deg_j:     torch.Tensor,
                embs_i:    torch.Tensor,   # (E, K, dim)
                embs_j:    torch.Tensor,
                times_i:   torch.Tensor,   # (E, K)
                times_j:   torch.Tensor,
                edge_emb:  torch.Tensor,   # (E, heads, head_dim)
                time_emb:  torch.Tensor,
                index:     torch.Tensor,   # (E,) target indices for softmax
                ) -> torch.Tensor:

        E = x_i.size(0)

        # Enrich keys
        K_rich = K_j + edge_emb + time_emb

        # TGATv2 score
        Q_flat = Q_i.reshape(E, self.dim)
        K_flat = K_rich.reshape(E, self.dim)
        scores = self.attn_scorer(
            torch.cat([Q_flat, K_flat], dim=-1)
        )                                                            # (E, heads)

        # Trust score
        trust = self.trust(
            h_i    = mem_val_i,
            h_j    = mem_val_j,
            deg_i  = deg_i,
            deg_j  = deg_j,
            embs_i = embs_i,
            embs_j = embs_j,
            t_i    = times_i,
            t_j    = times_j,
        )                                                            # (E,) in (0,2)
        self._trust_cache = trust.detach()

        # Multiply before softmax so normalisation reflects trust-adjusted importance
        scores = scores * trust.unsqueeze(-1)                        # (E, heads)
        scores = softmax(scores, index)
        scores = self.dropout(scores)

        return V_j * scores.unsqueeze(-1)                           # (E, h, hd)


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  EnhancedTemporalGNN  — full model
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedTemporalGNN(nn.Module):
    """
    Full temporal GNN with K-view trajectory trust for representation learning.

    Expected batch fields:
        batch.x          (N, in_dim)    node features
        batch.edge_index (2, E)         directed edges [src, dst]
        batch.msg        (E, edge_dim)  edge / message features
        batch.t          (E,)           edge timestamps

    Returns:
        x  (N, hidden_dim)  node representations
    """
    def __init__(
        self,
        num_nodes:       int,
        in_dim:          int,
        edge_dim:        int,
        hidden_dim:      int,
        num_layers:      int   = 2,
        heads:           int   = 8,
        dropout:         float = 0.1,
        window:          int   = 8,
        t2v_dim:         int   = 16,
        memory_momentum: float = 0.9,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes  = num_nodes
        self.window     = window

        # Input projections
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_enc   = nn.Linear(edge_dim, hidden_dim)
        self.time_enc   = TimeEncoder(hidden_dim)

        # TransformerConv snapshot encoder: produces embeddings stored in EvolutionBank
        self.snapshot_gnn = SnapshotGNN(hidden_dim, heads=heads, dropout=dropout)

        # Persistent state
        self.memory   = RecurrentMemory(num_nodes, hidden_dim, momentum=memory_momentum)
        self.evo_bank = EvolutionBank(num_nodes, hidden_dim, window=window)

        # Main message-passing layers
        self.layers = nn.ModuleList([
            TemporalTrustAttention(
                hidden_dim, heads=heads, dropout=dropout,
                window=window, t2v_dim=t2v_dim,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    # --------------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # --------------------------------------------------------------------------
    def forward(self, batch) -> torch.Tensor:
        N          = batch.x.size(0)
        device     = batch.x.device
        edge_index = batch.edge_index
        src, dst   = edge_index

        # ── Encode inputs ─────────────────────────────────────────────────────
        x        = self.input_proj(batch.x)                         # (N, hidden)
        edge_emb = self.edge_enc(batch.msg)                         # (E, hidden)

        # ── Node degrees ──────────────────────────────────────────────────────
        deg = degree(dst, num_nodes=N, dtype=torch.float).clamp(min=1.0)

        # ── Temporal encoding ─────────────────────────────────────────────────
        node_last_t = torch.zeros(N, device=device)
        node_last_t.scatter_reduce_(
            0, src, batch.t.float(), reduce='amax', include_self=True
        )
        t_rel    = batch.t.float() - node_last_t[src]
        time_emb = self.time_enc(batch.t, t_rel)                    # (E, hidden)

        # ── Compute snapshot embedding for EvolutionBank ──────────────────────
        # TransformerConv operates directly on node features and edge_attr.
        # No time signal here — time is injected later in TemporalPositionalEncoding.
        snapshot_emb = self.snapshot_gnn(x, edge_index, edge_emb)   # (N, hidden)

        # ── Message passing layers ─────────────────────────────────────────────
        for layer in self.layers:
            x = layer(x, edge_index, edge_emb, time_emb,
                      self.memory, self.evo_bank, deg)

        x = self.final_norm(x)

        # ── Update persistent state ───────────────────────────────────────────
        nidx      = torch.arange(N, device=device)
        current_t = batch.t.float().max().expand(N)

        # Write snapshot to EvolutionBank (the rich contextual embedding)
        self.evo_bank.write(nidx, snapshot_emb, current_t)
        # Update GRU memory
        self.memory.write(nidx, x)

        return x

    # --------------------------------------------------------------------------
    def get_trust_scores(self, layer_idx: int = -1) -> Optional[torch.Tensor]:
        """Cached trust scores (E,) in (0, 2) from last forward()."""
        return self.layers[layer_idx]._trust_cache

    @torch.no_grad()
    def reset_memory(self):
        self.memory.hidden.zero_()
        self.memory.variance.fill_(0.1)
        self.evo_bank.reset()


# ═══════════════════════════════════════════════════════════════════════════════
# Quick sanity check
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import types

    torch.manual_seed(42)

    N, E      = 50, 120
    in_dim    = 32
    edge_dim  = 16
    hidden    = 64
    num_nodes = 100
    K         = 8

    def make_batch(t_offset=0.0):
        return types.SimpleNamespace(
            x          = torch.randn(N, in_dim),
            edge_index = torch.randint(0, N, (2, E)),
            msg        = torch.randn(E, edge_dim),
            t          = torch.rand(E) * 1000 + t_offset,
        )

    model = EnhancedTemporalGNN(
        num_nodes=num_nodes, in_dim=in_dim, edge_dim=edge_dim,
        hidden_dim=hidden, num_layers=2, heads=4,
        window=K, t2v_dim=16,
    )

    for step in range(3):
        emb   = model(make_batch(step * 1000))
        trust = model.get_trust_scores()
        print(f"[pass {step+1}] emb: {emb.shape} | "
              f"trust: [{trust.min():.4f}, {trust.max():.4f}] "
              f"mean={trust.mean():.4f}")
        assert trust.min() > 0.0 and trust.max() < 2.0

    print("\nAll checks passed. Trust is in (0, 2).")