"""
Temporal Trust GNN for Unsupervised Anomaly Detection
=====================================================
Produces node representations for downstream contrastive learning.
No labels required — fully unsupervised.

Message Aggregation (TemporalTrustAttention):

  [1] CROSS-SOURCE ATTENTION
      Q comes from EDGE embeddings  (what does this interaction ask for?)
      K comes from NODE features x  (who is this neighbour?)
      The same neighbour j is weighted differently depending on the
      type/context of the edge, making attention interaction-aware.

  [2] RoPE ON MESSAGES  (MessageRoPE)
      After scoring, the value message V_j is rotated by its edge
      timestamp using continuous-time RoPE BEFORE aggregation.
      Time is therefore baked into the message CONTENT, not just
      the score — messages from different times aggregate to different
      vectors even with identical node features.

  [3] SIMILARITY-BASED TRUST  (StructuralTrustModule)
      trust(i->j) in (0, 2):
        trust > 1 : i and j are structurally similar -> amplify
        trust < 1 : i and j are dissimilar           -> attenuate
      Both normal<->normal and anomalous<->anomalous get high trust.
      The model self-segregates: each node learns from its own "type".
      This naturally separates normal and anomalous representations.

  [4] TEMPORAL MESSAGE BANK  (TemporalMessageBank) — NEW
      A per-node buffer stores the last M aggregated messages.
      After each aggregation, cross-attention reads over this buffer:
        Q = current aggregated message
        K/V = past M stored messages
      Output: temporally-smoothed message that blends current with
      historical incoming signals. Detects sudden anomalous changes
      in what a node is receiving (erratic = anomalous influence).

Trust signals (StructuralTrustModule):
  [A] cos(h_i, h_j)              current GRU state similarity
  [B] degree profile similarity  structural role similarity
  [C] K-view trajectory sim      RoPE+Time2Vec encoded snapshot comparison
      trust = 1 + tanh(MLP([A, B, C]))   in (0, 2)
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
# 8.  MessageRoPE  — applies continuous-time RoPE directly to message vectors
#
#  Unlike the TemporalPositionalEncoding used in the trust module (which
#  operates on stored K-view snapshots), this module applies RoPE to the
#  LIVE message vector being passed from j to i along an edge, using the
#  edge's own timestamp as the rotation angle.
#
#  Why apply RoPE to messages (not just to queries/keys)?
#  ────────────────────────────────────────────────────────
#  In standard transformer attention, RoPE is applied to Q and K so that
#  the dot product Q·K reflects relative position.  Here we go further:
#  the VALUE (message content) is also rotated by the edge timestamp.
#  This means:
#    - Two messages with the same content but different timestamps will
#      aggregate to DIFFERENT vectors at the target node.
#    - The target node's representation is intrinsically time-aware,
#      not just score-aware.
#    - Recent messages occupy a different angular region from old ones,
#      so the GRU memory update naturally distinguishes temporal recency.
#
#  Shared log_timescales with the trust module's TemporalPositionalEncoding
#  are NOT used — this module has its own learned timescales tuned for
#  the message aggregation regime (shorter timescales work better here
#  since edge timestamps are absolute, not relative gaps).
# ═══════════════════════════════════════════════════════════════════════════════

class MessageRoPE(nn.Module):
    """
    Applies continuous-time RoPE to a batch of message vectors.

    Rotates pairs of dimensions by angles proportional to the edge timestamp:
        theta_d = t_edge / exp(log_timescale_d)
        msg[2d]   ->  msg[2d]   * cos(theta_d) - msg[2d+1] * sin(theta_d)
        msg[2d+1] ->  msg[2d]   * sin(theta_d) + msg[2d+1] * cos(theta_d)

    The dot product between two RoPE-encoded messages naturally reflects
    their temporal distance: same-time messages stay aligned, cross-time
    messages rotate apart in proportion to the timestamp gap.
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        # Shorter timescales than K-view RoPE — edge-level granularity
        self.log_timescales = nn.Parameter(
            torch.linspace(0, 6, dim // 2)    # timescales: 1 → e^6 ≈ 400
        )

    def forward(self,
                msg: torch.Tensor,   # (E, dim)
                t:   torch.Tensor,   # (E,)   edge timestamps
                ) -> torch.Tensor:   # (E, dim)
        timescales = self.log_timescales.exp()              # (D//2,)
        angles     = t.float().unsqueeze(-1) / timescales  # (E, D//2)

        cos_a = angles.cos()   # (E, D//2)
        sin_a = angles.sin()

        m_even = msg[:, 0::2]  # (E, D//2)
        m_odd  = msg[:, 1::2]

        rot_even = m_even * cos_a - m_odd * sin_a
        rot_odd  = m_even * sin_a + m_odd * cos_a

        # Interleave back to (E, D)
        rotated = torch.stack([rot_even, rot_odd], dim=-1)  # (E, D//2, 2)
        return rotated.reshape(msg.shape)


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  TemporalMessageBank  — NEW ADVANCED MECHANISM
#
#  Standard message passing aggregates ALL neighbours in one flat sum/mean.
#  This loses two things:
#    1. TEMPORAL ORDER: messages from t=10 and t=1000 are mixed equally.
#    2. TEMPORAL CONTEXT: the node has no memory of what it received before.
#
#  The TemporalMessageBank fixes both:
#    - After each forward pass, the aggregated message vector (already
#      RoPE-encoded) is stored in a per-node circular buffer of size M.
#    - Before the final update, a cross-attention reads over this buffer:
#        Query  = current aggregated message  (what I just received)
#        Key/V  = past M buffered messages    (what I've received before)
#      This lets the node ask: "is what I'm receiving NOW consistent with
#      my recent message history, or is it a sudden anomalous change?"
#    - The attention output is a temporally-smoothed message that blends
#      current and historical incoming signals.
#
#  Effect on anomaly detection:
#    Normal nodes receive consistent messages over time → strong cross-attn
#    Anomalous nodes receive erratic messages           → weak cross-attn
#    Central node near anomaly gets a divergent current msg vs history
#    → the bank effectively provides a "temporal message baseline"
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalMessageBank(nn.Module):
    """
    Per-node circular buffer of past aggregated messages with
    cross-attention retrieval.

    Args:
        num_nodes : total nodes in the graph
        dim       : message / hidden dimension
        bank_size : M, number of past messages to store per node
    """
    def __init__(self, num_nodes: int, dim: int, bank_size: int = 8):
        super().__init__()
        self.bank_size = bank_size
        self.dim       = dim
        # Stored past messages per node: (num_nodes, M, dim)
        self.register_buffer("bank", torch.zeros(num_nodes, bank_size, dim))
        self.register_buffer("ptr",  torch.zeros(num_nodes, dtype=torch.long))

        # Cross-attention: current msg as Q, past bank as K and V
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.scale  = dim ** -0.5
        self.out_proj = nn.Linear(dim, dim)
        self.norm     = nn.LayerNorm(dim)

    @torch.no_grad()
    def write(self, idx: torch.Tensor, msg: torch.Tensor):
        """Write current aggregated message into the circular buffer."""
        for b in range(idx.size(0)):
            node = int(idx[b])
            p    = int(self.ptr[node]) % self.bank_size
            self.bank[node, p] = msg[b].detach()
            self.ptr[node] += 1

    def read_and_attend(self,
                        idx:     torch.Tensor,   # (N,)
                        cur_msg: torch.Tensor,   # (N, dim)
                        ) -> torch.Tensor:       # (N, dim)
        """
        Cross-attention: current message queries the historical bank.

        Q = Linear(cur_msg)          (N, dim)
        K = Linear(bank[idx])        (N, M, dim)
        V = Linear(bank[idx])        (N, M, dim)

        output = softmax(Q K^T / sqrt(d)) V    (N, dim)

        Then blend: alpha * output + (1-alpha) * cur_msg
        where alpha is learned (starts near 0.5).
        """
        past = self.bank[idx]                                        # (N, M, dim)
        N, M, D = past.shape

        Q  = self.q_proj(cur_msg).unsqueeze(1)                       # (N, 1, dim)
        K  = self.k_proj(past)                                       # (N, M, dim)
        V  = self.v_proj(past)

        # Scaled dot-product attention over M past messages
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale         # (N, 1, M)
        attn = F.softmax(attn, dim=-1)
        ctx  = torch.bmm(attn, V).squeeze(1)                        # (N, dim)

        # Residual: blend retrieved history with current message
        out = self.out_proj(ctx)
        return self.norm(cur_msg + out)                              # (N, dim)

    @torch.no_grad()
    def reset(self):
        self.bank.zero_()
        self.ptr.zero_()


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  TemporalTrustAttention  (one message-passing layer)
#
#  Four upgrades vs the previous version:
#
#  [1] CROSS-SOURCE ATTENTION
#      Q comes from EDGE embeddings  (what is this interaction asking for?)
#      K comes from NODE features x  (who is this neighbour?)
#      This makes the attention interaction-aware:  the same neighbour j
#      will be weighted differently depending on the TYPE of edge (e.g.
#      a payment edge vs a message edge queries j differently).
#
#  [2] RoPE ON MESSAGES  (MessageRoPE)
#      After computing V_j (the value to send), apply RoPE with the edge
#      timestamp BEFORE aggregation.  So the aggregated sum at node i is
#      a timestamp-rotated superposition of neighbour values — time is
#      baked into the message content, not just the score.
#
#  [3] SIMILARITY-BASED TRUST  (StructuralTrustModule, unchanged formula)
#      Trust(i→j) ∈ (0,2) — amplifies SIMILAR nodes (normal↔normal or
#      anomalous↔anomalous) and attenuates DISSIMILAR ones.
#      This means the model self-segregates: nodes learn mostly from
#      their own "type", which separates the representation spaces of
#      normal and anomalous nodes without needing any labels.
#
#  [4] TEMPORAL MESSAGE BANK  (TemporalMessageBank)
#      After aggregation, cross-attend over the node's M most recently
#      stored aggregated messages.  Provides a temporal context window
#      over received messages, enabling the node to detect sudden changes
#      in what it is receiving (a hallmark of anomalous influence).
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalTrustAttention(MessagePassing):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1,
                 window: int = 8, t2v_dim: int = 16,
                 num_nodes: int = 0, bank_size: int = 8):
        super().__init__(aggr="add", node_dim=0)
        assert dim % heads == 0
        self.dim      = dim
        self.heads    = heads
        self.head_dim = dim // heads

        # ── Cross-source projections ──────────────────────────────────────────
        # Q from edge embedding (interaction type)
        self.q_edge_proj = nn.Linear(dim, dim)
        # K from node features (neighbour identity)
        self.k_node_proj = nn.Linear(dim, dim)
        # V from node features (what neighbour sends)
        self.v_proj      = nn.Linear(dim, dim)
        self.o_proj      = nn.Linear(dim, dim)

        # Scale factor for dot-product attention
        self.scale = self.head_dim ** -0.5

        # ── RoPE on messages ──────────────────────────────────────────────────
        self.msg_rope = MessageRoPE(dim)

        # ── Trust ─────────────────────────────────────────────────────────────
        self.trust   = StructuralTrustModule(dim, window=window, t2v_dim=t2v_dim)

        # ── Temporal message bank ─────────────────────────────────────────────
        # Only instantiated if num_nodes > 0
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
                edge_emb:   torch.Tensor,       # (E, dim)  raw (not split)
                time_emb:   torch.Tensor,       # (E, dim)  raw
                edge_t:     torch.Tensor,       # (E,)      edge timestamps
                memory:     RecurrentMemory,
                evo_bank:   EvolutionBank,
                deg:        torch.Tensor,       # (N,)
                ) -> torch.Tensor:

        N    = x.size(0)
        nidx = torch.arange(N, device=x.device)

        # ── Cross-source Q, K, V ──────────────────────────────────────────────
        # Q: from edge embedding — per edge, split into heads
        Q_edge = self.q_edge_proj(edge_emb).view(-1, self.heads, self.head_dim)
        # K: from node features — per node, split into heads
        K_node = self.k_node_proj(x).view(N, self.heads, self.head_dim)
        # V: from node features — per node, split into heads
        V_node = self.v_proj(x).view(N, self.heads, self.head_dim)

        mem_vals, _ = memory.read(nidx)
        embs_all, times_all = evo_bank.read(nidx)

        out = self.propagate(
            edge_index,
            x=x,
            Q_edge=Q_edge,         # (E, heads, hd) — already per-edge
            K_node=K_node,         # (N, heads, hd) — will be indexed as K_node_j
            V_node=V_node,         # (N, heads, hd) — will be indexed as V_node_j
            mem_val=mem_vals,
            deg=deg,
            embs=embs_all,
            times=times_all,
            edge_t=edge_t,         # (E,) raw timestamps for MessageRoPE
            size=(N, N),
        )
        out = out.view(N, self.dim)                                  # (N, dim)

        # ── Temporal Message Bank: retrieve temporal context ──────────────────
        if self.msg_bank is not None:
            out = self.msg_bank.read_and_attend(nidx, out)          # (N, dim)
            self.msg_bank.write(nidx, out)

        # ── Fuse with GRU memory + gated residual ─────────────────────────────
        h   = memory.get_hidden(nidx)
        out = self.fusion(torch.cat([out, h], dim=-1))
        out = self.o_proj(out)

        g = torch.sigmoid(self.gate(x))
        return self.norm(g * x + (1 - g) * out)

    # --------------------------------------------------------------------------
    def message(self,
                x_i:        torch.Tensor,   # (E, dim)   central node features
                x_j:        torch.Tensor,   # (E, dim)   neighbour features
                Q_edge:     torch.Tensor,   # (E, heads, head_dim)  from edge
                K_node_j:   torch.Tensor,   # (E, heads, head_dim)  from node j
                V_node_j:   torch.Tensor,   # (E, heads, head_dim)  from node j
                mem_val_i:  torch.Tensor,   # (E, dim)
                mem_val_j:  torch.Tensor,
                deg_i:      torch.Tensor,   # (E,)
                deg_j:      torch.Tensor,
                embs_i:     torch.Tensor,   # (E, K, dim)
                embs_j:     torch.Tensor,
                times_i:    torch.Tensor,   # (E, K)
                times_j:    torch.Tensor,
                edge_t:     torch.Tensor,   # (E,)   edge timestamps
                index:      torch.Tensor,   # (E,)   target indices for softmax
                ) -> torch.Tensor:

        E = x_i.size(0)

        # ── [1] Cross-source attention score ─────────────────────────────────
        # Q from edge embedding asks: "given this interaction, how relevant is j?"
        # K from node j answers:     "here is what I am structurally"
        # Dot product: (E, heads, hd) x (E, heads, hd) -> (E, heads)
        scores = (Q_edge * K_node_j).sum(-1) * self.scale            # (E, heads)

        # ── [2] Similarity-based trust ────────────────────────────────────────
        # trust(i→j) ∈ (0, 2): amplifies similar pairs (normal↔normal or
        # anomalous↔anomalous), attenuates dissimilar pairs.
        # Self-segregation: the model learns to cluster by type unsupervised.
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

        # Scale scores by trust BEFORE softmax:
        # - Similar j's (trust>1) get amplified relative scores
        # - Dissimilar j's (trust<1) get suppressed relative scores
        # After softmax, each central node i will draw mostly from
        # neighbours that are structurally similar to itself.
        scores = scores * trust.unsqueeze(-1)                        # (E, heads)
        scores = softmax(scores, index)                              # (E, heads)
        scores = self.dropout(scores)

        # ── [3] Compute value and apply RoPE with edge timestamp ──────────────
        # V comes from node j's features
        msg = V_node_j * scores.unsqueeze(-1)                        # (E, heads, hd)
        msg = msg.reshape(E, self.dim)                               # (E, dim)

        # Apply RoPE to the message using the edge's actual timestamp.
        # This rotates the message content by the time it was sent, so
        # the aggregated sum at i carries intrinsic temporal information.
        # Messages from t=100 and t=1000 will aggregate to different vectors
        # even if their content (V_j) was identical.
        msg = self.msg_rope(msg, edge_t)                             # (E, dim)

        return msg                                                   # (E, dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  EnhancedTemporalGNN  — full model
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
        bank_size:       int   = 8,
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
                num_nodes=num_nodes, bank_size=bank_size,
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

        # ── Temporal encoding (for TimeEncoder / GRU context) ─────────────────
        node_last_t = torch.zeros(N, device=device)
        node_last_t.scatter_reduce_(
            0, src, batch.t.float(), reduce='amax', include_self=True
        )
        t_rel    = batch.t.float() - node_last_t[src]
        time_emb = self.time_enc(batch.t, t_rel)                    # (E, hidden)

        # ── Compute snapshot embedding for EvolutionBank ──────────────────────
        # TransformerConv operates directly on node features and edge_attr.
        # No time signal here — time is injected in TemporalPositionalEncoding.
        snapshot_emb = self.snapshot_gnn(x, edge_index, edge_emb)   # (N, hidden)

        # ── Message passing layers ─────────────────────────────────────────────
        # Pass raw edge timestamps (batch.t) separately for MessageRoPE
        for layer in self.layers:
            x = layer(x, edge_index, edge_emb, time_emb,
                      batch.t.float(),                               # edge_t for RoPE
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
    )

    for step in range(3):
        emb   = model(make_batch(step * 1000))
        trust = model.get_trust_scores()
        print(f"[pass {step+1}] emb: {emb.shape} | "
              f"trust: [{trust.min():.4f}, {trust.max():.4f}] "
              f"mean={trust.mean():.4f}")
        assert trust.min() > 0.0 and trust.max() < 2.0

    print("\nAll checks passed. Trust is in (0, 2).")