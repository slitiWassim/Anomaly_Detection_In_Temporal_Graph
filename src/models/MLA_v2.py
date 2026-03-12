"""
Temporal Trust GNN for Unsupervised Anomaly Detection  (v3)
============================================================
Produces node representations for downstream contrastive learning.
No labels required — fully unsupervised.

─────────────────────────────────────────────────────────────────────────────
NEW IN v3 — MULTI-HEAD LATENT ATTENTION (MLA)
─────────────────────────────────────────────────────────────────────────────

  Standard MHA (v2) has three independent projections:
      Q: Linear(dim, dim)           ← D² params
      K: Linear(2*dim, dim)         ← 2D² params
      V: Linear(2*dim, dim)         ← 2D² params
      Total: 5D² projection params

  MLA replaces them with three compounding ideas:

  ┌────────────────────────────────────────────────────────────────────┐
  │ IDEA 1 — SHARED LOW-RANK LATENT BOTTLENECK (MLAProjection)        │
  │                                                                    │
  │  Q  path  (source node):                                           │
  │    x_src ──[W_Q_down]──► c_Q ──[LN]──► latent (dim/2)            │
  │                                  ├───[W_Q_nope_up]──► Q_nope      │
  │                                  └───[W_Q_rope_up]──► Q_rope      │
  │                                                                    │
  │  KV path  (target node + edge) — KEY INSIGHT: K and V share c_KV  │
  │    cat(x_dst, edge) ──[W_KV_down]──► c_KV ──[LN]──► latent       │
  │                                        ├──[W_K_nope_up]──► K_nope │
  │                                        ├──[W_K_rope_up]──► K_rope │
  │                                        └──[W_V_up]──────► V       │
  │                                                                    │
  │  With latent_dim = D/2:                                            │
  │    MLA params ≈ 3D²  vs  v2's 5D²  (40% reduction)                 │
  │  The shared bottleneck forces K and V to share structure:          │
  │  the "what to attend to" and "what to send" signals must both      │
  │  be explained by the same compressed representation of the         │
  │  target node — a strong inductive bias.                            │
  └────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────┐
  │ IDEA 2 — DECOUPLED RoPE  (HeadRoPE)                               │
  │                                                                    │
  │  Each head_dim splits into two non-overlapping sub-spaces:         │
  │    nope_hd = head_dim // 2   ← CONTENT dims   (no rotation)       │
  │    rope_hd = head_dim // 2   ← POSITIONAL dims (RoPE-rotated)     │
  │                                                                    │
  │  Q = [ Q_nope  |  Q_rope_enc ]   (E, H, head_dim)                 │
  │  K = [ K_nope  |  K_rope_enc ]   (E, H, head_dim)                 │
  │  V = [ V_full  ]                  no RoPE — V is semantic content  │
  │                                                                    │
  │  Attention score (additive decomposition):                         │
  │      score = (Q_nope · K_nope  +  Q_rope · K_rope) / √head_dim   │
  │               ───────────────────   ─────────────────             │
  │               content similarity    temporal similarity            │
  │                                                                    │
  │  WHY DECOUPLE:                                                     │
  │  Applying RoPE to the full head vector DISTORTS content distances: │
  │  two nodes with identical features but at different timestamps     │
  │  would have a very low raw score even if they are structurally     │
  │  identical.  Decoupling ensures:                                   │
  │    • Content similarity is TIME-INVARIANT (nope · nope)            │
  │    • Temporal alignment has its own dedicated sub-space (rope)     │
  │    • The two components contribute additively → interpretable      │
  └────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────┐
  │ IDEA 3 — LATENT LAYER NORM                                         │
  │                                                                    │
  │  LN(c_Q) and LN(c_KV) before decompression.                       │
  │  The down-projection has low rank → gradients can saturate or      │
  │  vanish in the bottleneck.  LayerNorm prevents this by keeping     │
  │  the latent distribution unit-normalised at each step.             │
  └────────────────────────────────────────────────────────────────────┘

  NEW IN v4 — OUTPUT-LEVEL TEMPORAL RoPE  (OutputRoPE)
  ─────────────────────────────────────────────────────
  In v3, RoPE was applied to Q_rope and K_rope inside MLAProjection so
  that the ATTENTION SCORE could capture temporal alignment between source
  and target.  However, the MESSAGE itself (the weighted sum of V) still
  carried no explicit temporal encoding — V was intentionally left without
  RoPE to preserve semantic content for aggregation.

  v4 adds a second, independent RoPE stage applied to the attention OUTPUT:

      msg_raw = (V * attn).reshape(E, dim)        ← weighted-value content
      msg_out = OutputRoPE( msg_raw, edge_t )     ← rotate by edge timestamp

  WHY APPLY RoPE TO THE OUTPUT (not just to Q/K)?
  ────────────────────────────────────────────────
  • Score-level RoPE (Q·K): controls WHICH neighbours contribute.
    Two messages from t=10 and t=1000 could still have the same score
    if the source node happens to be temporally aligned with both.
  • Output-level RoPE (msg): controls WHAT is contributed.
    Even if two messages receive the same attention weight, they will
    aggregate to DIFFERENT vectors at the target node because they are
    rotated to different angular regions of the representation space.
    The aggregated node embedding therefore retains a memory of WHEN
    each message arrived, not only WHO sent it.
  • The two RoPE stages are complementary and do not interfere:
      – MLAProjection's HeadRoPE: acts inside the head sub-space on
        (E, H, rope_hd) slices, tuned for score computation.
      – OutputRoPE: acts on the full flattened (E, dim) message vector
        with independently learned timescales tuned for message content.

  Preserved from v3 (UNCHANGED):
    • Q ← source node  |  K, V ← target node + edge features
    • MLA low-rank latent bottleneck
    • Decoupled nope/rope head sub-spaces
    • Latent LayerNorm
    • LeakyReLU-gated softmax
    • Structural trust modulation  (StructuralTrustModule)
    • Temporal Message Bank         (TemporalMessageBank)
    • GRU memory fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.utils import softmax, degree


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Time2Vec  (unchanged)
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
# 2.  TimeEncoder  (unchanged)
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
# 3.  RecurrentMemory  (unchanged)
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
# 4.  SnapshotGNN  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class SnapshotGNN(nn.Module):
    """
    Transformer-based snapshot encoder using PyG's TransformerConv.
    Produces a rich node embedding from node features and edge attributes.
    """
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0

        self.conv1 = TransformerConv(
            in_channels=dim, out_channels=dim // heads, heads=heads,
            dropout=dropout, edge_dim=dim, concat=True, beta=True,
        )
        self.norm1 = nn.LayerNorm(dim)

        self.conv2 = TransformerConv(
            in_channels=dim, out_channels=dim // heads, heads=heads,
            dropout=dropout, edge_dim=dim, concat=True, beta=True,
        )
        self.norm2 = nn.LayerNorm(dim)

        self.out_proj = nn.Linear(dim, dim)
        self.gate     = nn.Linear(dim, 1)

    def forward(self,
                x:          torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr:  torch.Tensor,
                ) -> torch.Tensor:
        h  = self.norm1(F.gelu(self.conv1(x, edge_index, edge_attr)))
        h2 = self.norm2(F.gelu(self.conv2(h, edge_index, edge_attr)))
        h  = h + h2
        out = self.out_proj(h)
        g   = torch.sigmoid(self.gate(x))
        return g * x + (1 - g) * out


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  EvolutionBank  (unchanged)
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
        return self.bank[idx], self.times[idx]

    @torch.no_grad()
    def write(self, idx: torch.Tensor, emb: torch.Tensor, t: torch.Tensor):
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
# 6.  TemporalPositionalEncoding  (unchanged — used by StructuralTrustModule)
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalPositionalEncoding(nn.Module):
    """RoPE + Time2Vec positional encoding for K snapshot embeddings."""
    def __init__(self, dim: int, t2v_dim: int = 16):
        super().__init__()
        assert dim % 2 == 0
        self.dim      = dim
        self.t2v_dim  = t2v_dim
        self.log_timescales = nn.Parameter(torch.linspace(0, 9, dim // 2))
        self.t2v      = Time2Vec(t2v_dim)
        self.t2v_proj = nn.Linear(t2v_dim, dim)

    def _apply_rope(self, embs: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        B, K, D    = embs.shape
        timescales = self.log_timescales.exp()
        angles     = times.unsqueeze(-1) / timescales.unsqueeze(0).unsqueeze(0)
        cos_a, sin_a = angles.cos(), angles.sin()
        e_even, e_odd = embs[..., 0::2], embs[..., 1::2]
        rot_even = e_even * cos_a - e_odd * sin_a
        rot_odd  = e_even * sin_a + e_odd * cos_a
        return torch.stack([rot_even, rot_odd], dim=-1).reshape(B, K, D)

    def forward(self, embs: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        B, K, D   = embs.shape
        rope_embs = self._apply_rope(embs, times)
        t_flat    = times.reshape(B * K)
        t2v_enc   = self.t2v(t_flat).view(B, K, self.t2v_dim)
        t2v_add   = self.t2v_proj(t2v_enc)
        return rope_embs + t2v_add


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  StructuralTrustModule  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class StructuralTrustModule(nn.Module):
    def __init__(self, dim: int, window: int = 8, t2v_dim: int = 16):
        super().__init__()
        self.window  = window
        self.time_pe = TemporalPositionalEncoding(dim, t2v_dim=t2v_dim)
        h = max(dim // 4, 16)
        self.trust_mlp = nn.Sequential(
            nn.Linear(3, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, 1),
        )

    def forward(self, h_i, h_j, deg_i, deg_j,
                embs_i, embs_j, t_i, t_j, eps=1e-6) -> torch.Tensor:
        emb_cos = (F.normalize(h_i, dim=-1, eps=eps)
                   * F.normalize(h_j, dim=-1, eps=eps)
                   ).sum(-1, keepdim=True)

        di, dj  = deg_i.float(), deg_j.float()
        deg_sim = (1.0 - (di - dj).abs() / (di + dj + eps)).unsqueeze(-1)

        views_i  = self.time_pe(embs_i, t_i)
        views_j  = self.time_pe(embs_j, t_j)
        vi_n     = F.normalize(views_i, dim=-1, eps=eps)
        vj_n     = F.normalize(views_j, dim=-1, eps=eps)
        traj_sim = (vi_n * vj_n).sum(-1).mean(-1, keepdim=True)

        features    = torch.cat([emb_cos, deg_sim, traj_sim], dim=-1)
        trust_logit = self.trust_mlp(features).squeeze(-1)
        return 1.0 + torch.tanh(trust_logit)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  TemporalMessageBank  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalMessageBank(nn.Module):
    """Per-node circular buffer with cross-attention retrieval."""
    def __init__(self, num_nodes: int, dim: int, bank_size: int = 8):
        super().__init__()
        self.bank_size = bank_size
        self.dim       = dim
        self.register_buffer("bank", torch.zeros(num_nodes, bank_size, dim))
        self.register_buffer("ptr",  torch.zeros(num_nodes, dtype=torch.long))

        self.q_proj   = nn.Linear(dim, dim)
        self.k_proj   = nn.Linear(dim, dim)
        self.v_proj   = nn.Linear(dim, dim)
        self.scale    = dim ** -0.5
        self.out_proj = nn.Linear(dim, dim)
        self.norm     = nn.LayerNorm(dim)

    @torch.no_grad()
    def write(self, idx: torch.Tensor, msg: torch.Tensor):
        for b in range(idx.size(0)):
            node = int(idx[b])
            p    = int(self.ptr[node]) % self.bank_size
            self.bank[node, p] = msg[b].detach()
            self.ptr[node] += 1

    def read_and_attend(self, idx: torch.Tensor,
                        cur_msg: torch.Tensor) -> torch.Tensor:
        past = self.bank[idx]                                    # (N, M, dim)
        N, M, D = past.shape
        Q    = self.q_proj(cur_msg).unsqueeze(1)                 # (N, 1, dim)
        K    = self.k_proj(past)                                 # (N, M, dim)
        V    = self.v_proj(past)
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale     # (N, 1, M)
        attn = F.softmax(attn, dim=-1)
        ctx  = torch.bmm(attn, V).squeeze(1)                    # (N, dim)
        return self.norm(cur_msg + self.out_proj(ctx))

    @torch.no_grad()
    def reset(self):
        self.bank.zero_()
        self.ptr.zero_()


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  HeadRoPE  — NEW  (per-head RoPE on the positional sub-dimension)
#
#  Unlike the flat MessageRoPE used in v1/v2 (which operated on the full dim),
#  HeadRoPE is aware of the head structure and applies rotation only to the
#  rope_hd-sized sub-space of each head independently.
#
#  Input shape:  (E, H, rope_hd)   — the positional slice of each head
#  Output shape: (E, H, rope_hd)   — rotated by edge timestamp
#
#  The learned log_timescales are shared across heads (one set of
#  rope_hd/2 timescales), keeping the positional space consistent.
#  If head-specific timescales are desired they can be expanded to
#  (H, rope_hd/2) — a straightforward extension.
# ═══════════════════════════════════════════════════════════════════════════════

class HeadRoPE(nn.Module):
    """
    Continuous-time RoPE operating on the positional sub-dimension of
    attention heads.  Only the rope_hd portion of each head is rotated;
    the nope_hd content portion is never touched.

    Rotation formula for pair (2d, 2d+1) of the rope sub-space:
        θ_d  = t_edge / exp(log_timescale_d)
        x[2d]   → x[2d]   * cos(θ) − x[2d+1] * sin(θ)
        x[2d+1] → x[2d]   * sin(θ) + x[2d+1] * cos(θ)
    """
    def __init__(self, rope_hd: int):
        super().__init__()
        assert rope_hd % 2 == 0, "rope_hd must be even"
        # Timescales for the rope sub-space: shorter range than trust-module
        # RoPE (edge-level timestamps, not snapshot gaps)
        self.log_timescales = nn.Parameter(
            torch.linspace(0, 6, rope_hd // 2)    # 1 → e^6 ≈ 400
        )

    def forward(self,
                x: torch.Tensor,   # (E, H, rope_hd)
                t: torch.Tensor,   # (E,)
                ) -> torch.Tensor: # (E, H, rope_hd)
        E, H, D = x.shape
        timescales = self.log_timescales.exp()              # (D//2,)

        # angles: (E, D//2)  →  broadcast over H heads
        angles = t.float().unsqueeze(-1) / timescales       # (E, D//2)
        angles = angles.unsqueeze(1).expand(E, H, D // 2)  # (E, H, D//2)

        cos_a, sin_a = angles.cos(), angles.sin()

        x_even = x[..., 0::2]   # (E, H, D//2)
        x_odd  = x[..., 1::2]

        rot_even = x_even * cos_a - x_odd * sin_a
        rot_odd  = x_even * sin_a + x_odd * cos_a

        # Interleave pairs back: (E, H, D//2, 2) → (E, H, D)
        return torch.stack([rot_even, rot_odd], dim=-1).reshape(E, H, D)


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  OutputRoPE  — NEW in v4  (RoPE applied to the flat attention output)
#
#  Distinct from HeadRoPE in two ways:
#    • Input shape: (E, dim) — the fully flattened weighted-value message,
#      AFTER heads have been concatenated.  HeadRoPE works on (E, H, rope_hd).
#    • Purpose: encodes WHEN this message was produced into the message
#      CONTENT, so the aggregated node embedding retains temporal provenance.
#      HeadRoPE encodes time into Q/K for score computation; OutputRoPE
#      encodes time into the message vector for representation learning.
#    • Timescales: independently learned, initialised over a wider range
#      (0 → 8) than HeadRoPE (0 → 6) because the output message must
#      distinguish timestamps across the full graph history rather than just
#      within a single attention layer's score computation.
#
#  Rotation formula (same as all RoPE variants):
#      θ_d = t / exp(log_timescale_d)
#      out[2d]   = msg[2d]   * cos(θ_d) − msg[2d+1] * sin(θ_d)
#      out[2d+1] = msg[2d]   * sin(θ_d) + msg[2d+1] * cos(θ_d)
# ═══════════════════════════════════════════════════════════════════════════════

class OutputRoPE(nn.Module):
    """
    Continuous-time RoPE applied to the flat (E, dim) attention output vector.

    Encodes the edge timestamp directly into the message CONTENT so that
    messages arriving at different times produce different aggregate vectors
    at the target node even when their semantic content (V) is identical.

    This is the v4 contribution: temporal information is now present at
    BOTH the score level (HeadRoPE on Q/K) AND the message content level
    (OutputRoPE on the weighted-value output).

    Args:
        dim : must be even (full hidden dimension after head concatenation)
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for OutputRoPE"
        # Wider timescale range than HeadRoPE: message content needs to
        # distinguish timestamps across the full graph history.
        self.log_timescales = nn.Parameter(
            torch.linspace(0, 8, dim // 2)    # 1 → e^8 ≈ 2981
        )

    def forward(self,
                msg: torch.Tensor,   # (E, dim)  flattened attention output
                t:   torch.Tensor,   # (E,)      edge timestamps
                ) -> torch.Tensor:   # (E, dim)  temporally-encoded message
        timescales = self.log_timescales.exp()              # (dim//2,)
        angles     = t.float().unsqueeze(-1) / timescales  # (E, dim//2)

        cos_a, sin_a = angles.cos(), angles.sin()

        m_even = msg[:, 0::2]   # (E, dim//2)
        m_odd  = msg[:, 1::2]

        rot_even = m_even * cos_a - m_odd * sin_a
        rot_odd  = m_even * sin_a + m_odd * cos_a

        # Interleave pairs back: (E, dim//2, 2) → (E, dim)
        return torch.stack([rot_even, rot_odd], dim=-1).reshape(msg.shape)


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  MLAProjection  — (unchanged from v3)
#
#  Encapsulates the entire Q / KV latent projection pipeline.
#  Called once per forward pass with edge-level tensors; returns five
#  pre-built, per-head tensors ready for the attention score computation.
#
#  STEP-BY-STEP FORWARD PASS:
#
#  ── Q path (source node) ──────────────────────────────────────────────
#
#  Step 1 — Compress source features to Q latent
#    c_Q = W_Q_down( x_src )                                (E, latent_q)
#    c_Q = LayerNorm( c_Q )
#    Motivation: low-rank bottleneck distils the source node's most
#    relevant features for the query.  LayerNorm stabilises training.
#
#  Step 2 — Decompress into CONTENT (nope) and POSITIONAL (rope) parts
#    Q_nope = W_Q_nope_up( c_Q ).view(E, H, nope_hd)       content part
#    Q_rope = W_Q_rope_up( c_Q ).view(E, H, rope_hd)       positional part
#
#  Step 3 — Apply HeadRoPE to the positional part only
#    Q_rope = HeadRoPE( Q_rope, t )
#
#  ── KV path (target node + edge — SHARED LATENT) ──────────────────────
#
#  Step 4 — Compress target+edge to the SHARED KV latent
#    c_KV = W_KV_down( cat(x_dst, edge_emb) )              (E, latent_kv)
#    c_KV = LayerNorm( c_KV )
#    Motivation: K and V share this bottleneck, which forces them to
#    agree on what structural information is worth preserving — a strong
#    regularisation that reduces redundancy between K and V.
#
#  Step 5 — Decompress K (content + positional) from shared latent
#    K_nope = W_K_nope_up( c_KV ).view(E, H, nope_hd)
#    K_rope = W_K_rope_up( c_KV ).view(E, H, rope_hd)
#    K_rope = HeadRoPE( K_rope, t )
#
#  Step 6 — Decompress V from the same shared latent
#    V = W_V_up( c_KV ).view(E, H, head_dim)
#    No RoPE on V: V carries SEMANTIC CONTENT to be aggregated, not
#    positional information.  Rotating V would inject time-bias into the
#    aggregated representation, interfering with the trust module which
#    already handles structural/temporal weighting.
#
#  ── Output ────────────────────────────────────────────────────────────
#  Returns: Q_nope, Q_rope, K_nope, K_rope, V
#  These are consumed by TemporalTrustAttention.message() to compute:
#    score = (Q_nope · K_nope  +  Q_rope · K_rope) / sqrt(head_dim)
# ═══════════════════════════════════════════════════════════════════════════════

class MLAProjection(nn.Module):
    """
    Multi-Head Latent Attention projection module.

    Args:
        dim          : hidden dimension (must equal heads * head_dim)
        heads        : number of attention heads
        latent_q_dim : bottleneck width for the Q path  (default: dim // 2)
        latent_kv_dim: bottleneck width for the shared KV path (default: dim // 2)
        nope_hd      : content sub-space per head  (default: head_dim // 2)
        rope_hd      : positional sub-space per head (default: head_dim // 2)

    Constraint: nope_hd + rope_hd == head_dim == dim // heads
    """
    def __init__(self,
                 dim:           int,
                 heads:         int,
                 latent_q_dim:  int,
                 latent_kv_dim: int,
                 nope_hd:       int,
                 rope_hd:       int):
        super().__init__()
        self.heads    = heads
        self.nope_hd  = nope_hd
        self.rope_hd  = rope_hd
        self.head_dim = nope_hd + rope_hd   # must equal dim // heads

        assert self.head_dim * heads == dim, \
            f"nope_hd({nope_hd}) + rope_hd({rope_hd}) must equal dim//heads={dim//heads}"
        assert rope_hd % 2 == 0, "rope_hd must be even for RoPE"

        # ── Q path ────────────────────────────────────────────────────────────
        # Down-project source node features to low-rank Q latent
        self.q_down     = nn.Linear(dim, latent_q_dim, bias=False)
        self.q_norm     = nn.LayerNorm(latent_q_dim)
        # Decompress: content (nope) and positional (rope) branches
        self.q_nope_up  = nn.Linear(latent_q_dim, heads * nope_hd, bias=False)
        self.q_rope_up  = nn.Linear(latent_q_dim, heads * rope_hd, bias=False)

        # ── KV path (shared latent) ───────────────────────────────────────────
        # Down-project cat(x_dst, edge_emb) to shared low-rank KV latent
        self.kv_down    = nn.Linear(dim * 2, latent_kv_dim, bias=False)
        self.kv_norm    = nn.LayerNorm(latent_kv_dim)
        # K decompression: content and positional branches
        self.k_nope_up  = nn.Linear(latent_kv_dim, heads * nope_hd, bias=False)
        self.k_rope_up  = nn.Linear(latent_kv_dim, heads * rope_hd, bias=False)
        # V decompression: full head_dim (no split — no RoPE applied to V)
        self.v_up       = nn.Linear(latent_kv_dim, heads * self.head_dim, bias=False)

        # ── Decoupled RoPE ────────────────────────────────────────────────────
        # Separate HeadRoPE for Q and K so each learns its own timescales.
        # Q timescales capture "what I'm looking for now" (query recency).
        # K timescales capture "what this interaction offered" (key context).
        self.q_rope_enc = HeadRoPE(rope_hd)
        self.k_rope_enc = HeadRoPE(rope_hd)

    # --------------------------------------------------------------------------
    def forward(self,
                x_src:    torch.Tensor,   # (E, dim)  source node features
                x_dst:    torch.Tensor,   # (E, dim)  target node features
                edge_emb: torch.Tensor,   # (E, dim)  edge embeddings
                t:        torch.Tensor,   # (E,)      edge timestamps
                ) -> Tuple[torch.Tensor, ...]:
        """
        Returns Q_nope, Q_rope, K_nope, K_rope, V — all edge-level, per-head.

        Shapes:
            Q_nope, K_nope : (E, H, nope_hd)
            Q_rope, K_rope : (E, H, rope_hd)  — RoPE-encoded
            V              : (E, H, head_dim)  — no RoPE
        """
        E = x_src.size(0)

        # ── Step 1–3: Q path ──────────────────────────────────────────────────
        c_Q    = self.q_norm(self.q_down(x_src))                     # (E, lq)
        Q_nope = self.q_nope_up(c_Q).view(E, self.heads, self.nope_hd)
        Q_rope = self.q_rope_up(c_Q).view(E, self.heads, self.rope_hd)
        Q_rope = self.q_rope_enc(Q_rope, t)                          # RoPE-encoded

        # ── Step 4–6: KV path (shared latent) ────────────────────────────────
        # K and V both decompress from c_KV — forcing shared structure
        c_KV   = self.kv_norm(
                     self.kv_down(torch.cat([x_dst, edge_emb], dim=-1))
                 )                                                    # (E, lkv)
        K_nope = self.k_nope_up(c_KV).view(E, self.heads, self.nope_hd)
        K_rope = self.k_rope_up(c_KV).view(E, self.heads, self.rope_hd)
        K_rope = self.k_rope_enc(K_rope, t)                          # RoPE-encoded
        V      = self.v_up(c_KV).view(E, self.heads, self.head_dim)  # no RoPE

        return Q_nope, Q_rope, K_nope, K_rope, V


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  TemporalTrustAttention  — REVISED to use MLAProjection  (v3)
#
#  WHAT CHANGED vs v2:
#  ────────────────────────────────────────────────────────────────────────────
#  Replaced the three independent Linear projections (q_src_proj, k_tgt_proj,
#  v_tgt_proj) and three separate MessageRoPE modules with a single
#  MLAProjection block that implements all three MLA ideas together.
#
#  Attention score in message() now uses the additive decomposition:
#
#      score_content  = (Q_nope  · K_nope)  * scale   ← time-invariant
#      score_temporal = (Q_rope  · K_rope)  * scale   ← time-dependent
#      raw_score      = score_content + score_temporal
#
#  Everything downstream (trust multiplication, LeakyReLU-gated softmax,
#  value weighting, message bank, GRU fusion) is unchanged.
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalTrustAttention(MessagePassing):
    def __init__(self,
                 dim:           int,
                 heads:         int   = 8,
                 dropout:       float = 0.1,
                 window:        int   = 8,
                 t2v_dim:       int   = 16,
                 num_nodes:     int   = 0,
                 bank_size:     int   = 8,
                 leaky_slope:   float = 0.2,
                 latent_ratio:  float = 0.5,   # latent_dim = dim * latent_ratio
                 rope_ratio:    float = 0.5,   # rope_hd   = head_dim * rope_ratio
                 ):
        super().__init__(aggr="add", node_dim=0)
        assert dim % heads == 0
        self.dim         = dim
        self.heads       = heads
        self.head_dim    = dim // heads
        self.leaky_slope = leaky_slope

        # Derive latent and sub-space dimensions
        latent_q_dim  = max(int(dim * latent_ratio), heads)
        latent_kv_dim = max(int(dim * latent_ratio), heads)
        rope_hd       = max(int(self.head_dim * rope_ratio) // 2 * 2, 2)  # even
        nope_hd       = self.head_dim - rope_hd
        assert nope_hd > 0, \
            f"rope_ratio={rope_ratio} leaves no room for nope dimensions"

        # Scale factor for the FULL head dimension (nope + rope)
        self.scale = self.head_dim ** -0.5

        # ── MLA projection block (replaces q/k/v_proj + rope modules) ─────────
        self.mla = MLAProjection(
            dim           = dim,
            heads         = heads,
            latent_q_dim  = latent_q_dim,
            latent_kv_dim = latent_kv_dim,
            nope_hd       = nope_hd,
            rope_hd       = rope_hd,
        )

        # Output projection (applied after head concatenation)
        self.o_proj = nn.Linear(dim, dim)

        # ── Output-level temporal RoPE  (v4 addition) ────────────────────────
        # Applied to the flat (E, dim) weighted-value message inside message().
        # Encodes WHEN the message was produced into the message CONTENT,
        # complementing the score-level HeadRoPE already in MLAProjection.
        self.out_rope = OutputRoPE(dim)

        # ── Trust + message bank ──────────────────────────────────────────────
        self.trust = StructuralTrustModule(dim, window=window, t2v_dim=t2v_dim)
        self.msg_bank = (TemporalMessageBank(num_nodes, dim, bank_size=bank_size)
                         if num_nodes > 0 else None)

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
                edge_t:     torch.Tensor,       # (E,)  raw timestamps
                memory:     RecurrentMemory,
                evo_bank:   EvolutionBank,
                deg:        torch.Tensor,       # (N,)
                ) -> torch.Tensor:

        N        = x.size(0)
        src, dst = edge_index
        nidx     = torch.arange(N, device=x.device)
        t        = edge_t.float()

        # ── MLA: build all Q, K, V projections for every edge ─────────────────
        #
        #  x[src] (E, dim)       — source node features for Q path
        #  x[dst] (E, dim)       — target node features for KV path
        #  edge_emb (E, dim)     — edge context for KV path (fused with x_dst)
        #  t (E,)                — timestamps for HeadRoPE
        #
        #  After this call every tensor is (E, H, sub_hd) and already
        #  RoPE-encoded for the rope sub-spaces.
        Q_nope, Q_rope, K_nope, K_rope, V = self.mla(x[src], x[dst], edge_emb, t)

        # Load persistent state for all nodes (used by trust + fusion)
        mem_vals, _         = memory.read(nidx)
        embs_all, times_all = evo_bank.read(nidx)

        # ── Message passing ───────────────────────────────────────────────────
        # Q_nope/Q_rope/K_nope/K_rope/V are already edge-level (E, H, sub_hd).
        # PyG will pass them unchanged to message() since they have no _i/_j
        # counterpart requested.  Node-level tensors (x, mem_val, deg, embs,
        # times) will be gathered into _i (target) / _j (source) variants.
        out = self.propagate(
            edge_index,
            x        = x,            # (N, dim)    → x_i, x_j
            Q_nope   = Q_nope,       # (E, H, nope_hd) — edge-level, pass-through
            Q_rope   = Q_rope,       # (E, H, rope_hd)
            K_nope   = K_nope,       # (E, H, nope_hd)
            K_rope   = K_rope,       # (E, H, rope_hd)
            V        = V,            # (E, H, head_dim)
            mem_val  = mem_vals,     # (N, dim)    → mem_val_i, mem_val_j
            deg      = deg,          # (N,)        → deg_i, deg_j
            embs     = embs_all,     # (N, K, dim) → embs_i, embs_j
            times    = times_all,    # (N, K)      → times_i, times_j
            edge_t   = t,            # (E,)  — passed unchanged to message()
            size     = (N, N),
        )
        out = out.view(N, self.dim)                                  # (N, dim)

        # ── Temporal Message Bank ─────────────────────────────────────────────
        if self.msg_bank is not None:
            out = self.msg_bank.read_and_attend(nidx, out)
            self.msg_bank.write(nidx, out)

        # ── Fuse with GRU memory + gated residual ─────────────────────────────
        h   = memory.get_hidden(nidx)
        out = self.fusion(torch.cat([out, h], dim=-1))
        out = self.o_proj(out)

        g = torch.sigmoid(self.gate(x))
        return self.norm(g * x + (1 - g) * out)

    # --------------------------------------------------------------------------
    def message(self,
                x_i:       torch.Tensor,   # (E, dim)
                x_j:       torch.Tensor,   # (E, dim)
                Q_nope:    torch.Tensor,   # (E, H, nope_hd)  — edge-level, no _i/_j
                Q_rope:    torch.Tensor,   # (E, H, rope_hd)  — RoPE-encoded
                K_nope:    torch.Tensor,   # (E, H, nope_hd)
                K_rope:    torch.Tensor,   # (E, H, rope_hd)  — RoPE-encoded
                V:         torch.Tensor,   # (E, H, head_dim)
                mem_val_i: torch.Tensor,   # (E, dim)
                mem_val_j: torch.Tensor,
                deg_i:     torch.Tensor,   # (E,)
                deg_j:     torch.Tensor,
                embs_i:    torch.Tensor,   # (E, K, dim)
                embs_j:    torch.Tensor,
                times_i:   torch.Tensor,   # (E, K)
                times_j:   torch.Tensor,
                index:     torch.Tensor,   # (E,)  target node indices for softmax
                edge_t:    torch.Tensor,   # (E,)  edge timestamps for OutputRoPE
                ) -> torch.Tensor:         # (E, dim)

        E = x_i.size(0)

        # ── [1] Decoupled attention score ─────────────────────────────────────
        #
        #  The score decomposes additively into two independent terms:
        #
        #  Content term  (Q_nope · K_nope):
        #    Measures structural/feature similarity between the source node's
        #    query and the target+edge's key — independent of time.
        #    High when the source "needs" what the target "has" regardless of
        #    when the interaction happened.
        #
        #  Temporal term  (Q_rope · K_rope):
        #    Both vectors were RoPE-rotated by the edge timestamp.
        #    The dot product Q_rope · K_rope = ||Q||·||K||·cos(θ_q - θ_k)
        #    where θ encodes the timestamp.  Same-time interactions yield
        #    high scores; temporally distant interactions yield lower ones.
        #    This is the standard RoPE relative-position property applied
        #    to continuous timestamps instead of token indices.
        #
        #  Dividing by sqrt(head_dim) scales the combined score, where
        #  head_dim = nope_hd + rope_hd so that both terms are normalised
        #  relative to the total head capacity.
        #
        score_content  = (Q_nope * K_nope).sum(-1)  # (E, H)
        score_temporal = (Q_rope * K_rope).sum(-1)  # (E, H)
        raw_scores     = (score_content + score_temporal) * self.scale   # (E, H)

        # ── [2] Structural trust modulation ───────────────────────────────────
        #
        #  trust(i→j) ∈ (0, 2): amplifies structurally similar pairs
        #  (normal↔normal or anomalous↔anomalous) and attenuates dissimilar
        #  ones.  Applied before the activation so it shifts the pre-softmax
        #  logits: trusted neighbours get a stronger signal in both the
        #  positive (attend-to) and negative (suppress) directions.
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
        raw_scores = raw_scores * trust.unsqueeze(-1)               # (E, H)

        # ── [3] LeakyReLU-gated softmax ───────────────────────────────────────
        #
        #  Pass raw scores through LeakyReLU BEFORE the segment-wise softmax.
        #  • Positive scores: pass through unchanged → standard attention
        #  • Negative scores: multiplied by leaky_slope (0.2) → near-zero but
        #    still carry a gradient signal back through the network
        #  Effect: the model can learn to ACTIVELY SUPPRESS a neighbour
        #  (drive its score negative) rather than just ignoring it, and still
        #  receive a learning signal from that suppression decision.
        #  This is particularly useful for anomaly detection: the model learns
        #  to suppress anomalous neighbours explicitly.
        activated = F.leaky_relu(raw_scores,
                                 negative_slope=self.leaky_slope)   # (E, H)
        attn = softmax(activated, index)                            # (E, H)
        attn = self.dropout(attn)

        # ── [4] Weighted message + output-level temporal RoPE  (v4) ──────────
        #
        #  Step A — weighted value sum (same as v3):
        #    V carries the target+edge semantic content.
        #    Weight each head's value slice by its attention coefficient.
        msg = V * attn.unsqueeze(-1)                                # (E, H, hd)
        msg = msg.reshape(E, self.dim)                              # (E, dim)

        #  Step B — OutputRoPE: rotate the flat message by the edge timestamp.
        #
        #  This is the v4 addition.  After softmax+weighting, msg encodes
        #  WHAT the target node is sending.  OutputRoPE encodes WHEN it was
        #  sent by rotating pairs of dimensions by an angle θ_d = t/T_d.
        #
        #  Consequence for aggregation: two messages with identical semantic
        #  content (same V) but different timestamps will rotate to different
        #  angular positions in the representation space.  When multiple such
        #  messages are summed at the target node the aggregate is a
        #  superposition that carries temporal provenance — the receiving node
        #  "knows" not just what was sent but when, even after aggregation.
        #
        #  This operates on the full dim (not just rope_hd sub-space) with
        #  independently learned timescales, so it does not interfere with
        #  the head-level HeadRoPE used during score computation.
        msg = self.out_rope(msg, edge_t)                            # (E, dim)
        return msg


# ═══════════════════════════════════════════════════════════════════════════════
# 12.  EnhancedTemporalGNN  — full model
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

    MLA-specific hyper-parameters:
        latent_ratio : fraction of hidden_dim used for Q / KV latents
                       (default 0.5 → latent_dim = hidden_dim // 2)
        rope_ratio   : fraction of head_dim allocated to the RoPE sub-space
                       (default 0.5 → rope_hd = head_dim // 2, nope_hd = other half)
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
        bank_size:       int   = 8,
        memory_momentum: float = 0.9,
        leaky_slope:     float = 0.2,
        latent_ratio:    float = 0.5,
        rope_ratio:      float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes  = num_nodes
        self.window     = window

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_enc   = nn.Linear(edge_dim, hidden_dim)
        self.time_enc   = TimeEncoder(hidden_dim)

        self.snapshot_gnn = SnapshotGNN(hidden_dim, heads=heads, dropout=dropout)

        self.memory   = RecurrentMemory(num_nodes, hidden_dim, momentum=memory_momentum)
        self.evo_bank = EvolutionBank(num_nodes, hidden_dim, window=window)

        self.layers = nn.ModuleList([
            TemporalTrustAttention(
                hidden_dim, heads=heads, dropout=dropout,
                window=window, t2v_dim=t2v_dim,
                num_nodes=num_nodes, bank_size=bank_size,
                leaky_slope=leaky_slope,
                latent_ratio=latent_ratio,
                rope_ratio=rope_ratio,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch) -> torch.Tensor:
        N          = batch.x.size(0)
        device     = batch.x.device
        edge_index = batch.edge_index
        src, dst   = edge_index

        x        = self.input_proj(batch.x)
        edge_emb = self.edge_enc(batch.msg)
        deg      = degree(dst, num_nodes=N, dtype=torch.float).clamp(min=1.0)

        node_last_t = torch.zeros(N, device=device)
        node_last_t.scatter_reduce_(
            0, src, batch.t.float(), reduce='amax', include_self=True)
        t_rel    = batch.t.float() - node_last_t[src]
        time_emb = self.time_enc(batch.t, t_rel)

        snapshot_emb = self.snapshot_gnn(x, edge_index, edge_emb)

        for layer in self.layers:
            x = layer(x, edge_index, edge_emb, time_emb,
                      batch.t.float(), self.memory, self.evo_bank, deg)

        x = self.final_norm(x)

        nidx      = torch.arange(N, device=device)
        current_t = batch.t.float().max().expand(N)
        self.evo_bank.write(nidx, snapshot_emb, current_t)
        self.memory.write(nidx, x)

        return x

    def get_trust_scores(self, layer_idx: int = -1) -> Optional[torch.Tensor]:
        return self.layers[layer_idx]._trust_cache

    @torch.no_grad()
    def reset_memory(self):
        self.memory.hidden.zero_()
        self.memory.variance.fill_(0.1)
        self.evo_bank.reset()
        for layer in self.layers:
            if layer.msg_bank is not None:
                layer.msg_bank.reset()


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
        window=K, t2v_dim=16, bank_size=8,
        leaky_slope=0.2, latent_ratio=0.5, rope_ratio=0.5,
    )

    # Parameter count comparison
    total = sum(p.numel() for p in model.parameters())
    mla   = sum(p.numel() for layer in model.layers
                for p in layer.mla.parameters())
    print(f"Total params : {total:,}")
    print(f"MLA params   : {mla:,}  ({100*mla/total:.1f}% of total)\n")

    for step in range(3):
        emb   = model(make_batch(step * 1000))
        trust = model.get_trust_scores()
        print(f"[pass {step+1}] emb: {emb.shape} | "
              f"trust: [{trust.min():.4f}, {trust.max():.4f}] "
              f"mean={trust.mean():.4f}")
        assert trust.min() > 0.0 and trust.max() < 2.0

    print("\nAll checks passed.  Trust is in (0, 2).")