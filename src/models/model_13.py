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



import math
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
# 6.  TemporalPositionalEncoding  — per-head multi-resolution temporal encoding
#
#  Applies temporal encoding to K-view snapshot sequences for the trust module.
#
#  KEY IDEA: split the embedding into H temporal heads, each head covering a
#  DIFFERENT temporal resolution band. Head 0 uses fast-rotating RoPE
#  (captures short-range patterns, e.g. hourly), head H-1 uses slow-rotating
#  RoPE (captures long-range drift, e.g. weekly). This gives the model multiple
#  simultaneous temporal "lenses" operating in parallel, similar to how
#  multi-head attention gives multiple geometric viewpoints.
#
#  Per head h:
#    (a) RoPE with head-specific timescale band:
#            θ_{h,d} = t / T_{h,d}
#        T_{h,d} are sampled from a sub-range of [T_min, T_max] assigned to head h.
#        Head h rotates dimensions (h*hd) to ((h+1)*hd) with timescales in
#        [T_min + h*(T_max-T_min)/H, T_min + (h+1)*(T_max-T_min)/H].
#
#    (b) Multi-scale Fourier (additive, shared across heads):
#        Complete Fourier basis (sin + cos) over F log-uniform frequencies,
#        projected to dim. Captures absolute periodic patterns independently
#        of the RoPE rotation.
#
#    (c) Recency decay gate (per-head learned λ_h):
#        g_{h,k} = σ( -λ_h * Δt_k + b_h )
#        Different heads decay at different rates — fast heads strongly
#        discount old views, slow heads treat all views more equally.
#
#  Combined per head h, view k:
#      view_{h,k} = g_{h,k} * RoPE_h(emb_{h,k}, t_k) + Fourier_proj(t_k)[h*hd:(h+1)*hd]
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalPositionalEncoding(nn.Module):
    """
    Per-head multi-resolution temporal positional encoding for K-view sequences.

    Args:
        dim       : total embedding dimension (must be even, divisible by t_heads)
        t_heads   : number of temporal resolution heads
        n_fourier : Fourier frequency bands (sin+cos each)
        t_min_log : log of minimum timescale (default: 0 → T=1)
        t_max_log : log of maximum timescale (default: 9 → T≈8000)
    """
    def __init__(self, dim: int, t_heads: int = 4, n_fourier: int = 32,
                 t_min_log: float = 0.0, t_max_log: float = 9.0):
        super().__init__()
        assert dim % 2 == 0 and dim % t_heads == 0
        self.dim     = dim
        self.t_heads = t_heads
        self.head_d  = dim // t_heads   # dims per temporal head
        assert self.head_d % 2 == 0,   "head_d must be even for RoPE"

        # ── (a) Per-head RoPE timescales ──────────────────────────────────────
        # Each head h gets timescales from a dedicated sub-range of [t_min, t_max].
        # Head 0: fast (captures fine-grained recency)
        # Head H-1: slow (captures long-range historical trends)
        rope_params = []
        for h in range(t_heads):
            lo = t_min_log + h       * (t_max_log - t_min_log) / t_heads
            hi = t_min_log + (h + 1) * (t_max_log - t_min_log) / t_heads
            rope_params.append(torch.linspace(lo, hi, self.head_d // 2))
        # Stack to (t_heads, head_d//2), learned
        self.rope_log_ts = nn.Parameter(torch.stack(rope_params, dim=0))

        # ── (b) Multi-scale Fourier (shared, additive) ────────────────────────
        freqs = torch.logspace(-4, 1, n_fourier)
        self.register_buffer("fourier_freqs", freqs)
        self.fourier_proj = nn.Linear(2 * n_fourier, dim)

        # ── (c) Per-head recency decay gate ───────────────────────────────────
        # Each head has its own learned decay rate λ_h and bias b_h.
        # Fast heads (low h): init high λ → strong recency emphasis
        # Slow heads (high h): init low λ → weaker decay
        init_log_decay = torch.linspace(1.0, -1.0, t_heads)         # head 0 fastest
        self.log_decay  = nn.Parameter(init_log_decay)               # (t_heads,)
        self.decay_bias = nn.Parameter(torch.ones(t_heads))          # (t_heads,)

    # ──────────────────────────────────────────────────────────────────────────
    def _rope_per_head(self,
                       embs:  torch.Tensor,   # (B, K, dim)
                       times: torch.Tensor,   # (B, K)
                       ) -> torch.Tensor:     # (B, K, dim)
        """Apply per-head RoPE: each head h rotates its slice of the embedding
        with head-h-specific timescales."""
        B, K, D = embs.shape
        out_parts = []
        for h in range(self.t_heads):
            # Slice embedding for this head: (B, K, head_d)
            sl = embs[..., h * self.head_d : (h + 1) * self.head_d]
            ts  = self.rope_log_ts[h].exp()                          # (head_d//2,)
            ang = times.unsqueeze(-1) / ts                           # (B, K, hd//2)
            cos_a, sin_a = ang.cos(), ang.sin()
            e_even = sl[..., 0::2]
            e_odd  = sl[..., 1::2]
            rot = torch.stack(
                [e_even * cos_a - e_odd * sin_a,
                 e_even * sin_a + e_odd * cos_a], dim=-1
            ).reshape(B, K, self.head_d)
            out_parts.append(rot)
        return torch.cat(out_parts, dim=-1)                          # (B, K, dim)

    # ──────────────────────────────────────────────────────────────────────────
    def _fourier(self, times: torch.Tensor) -> torch.Tensor:
        """(B, K) -> (B, K, dim)"""
        B, K = times.shape
        t    = times.reshape(B * K, 1).float()
        phi  = t * self.fourier_freqs.unsqueeze(0)                   # (BK, F)
        feat = torch.cat([phi.sin(), phi.cos()], dim=-1)             # (BK, 2F)
        return self.fourier_proj(feat).view(B, K, self.dim)          # (B, K, D)

    # ──────────────────────────────────────────────────────────────────────────
    def _decay_gates(self, times: torch.Tensor) -> torch.Tensor:
        """(B, K) -> (B, K, dim)  — per-head gate broadcast across head dims"""
        t_max = times.max(dim=-1, keepdim=True).values               # (B, 1)
        delta = (t_max - times).clamp(min=0.0)                      # (B, K)
        lam   = self.log_decay.exp()                                 # (t_heads,)
        bias  = self.decay_bias                                      # (t_heads,)
        # gate per head: (B, K, t_heads)
        gate  = torch.sigmoid(
            -lam.unsqueeze(0).unsqueeze(0) * delta.unsqueeze(-1)
            + bias.unsqueeze(0).unsqueeze(0)
        )
        # Broadcast to full dim: each head's gate applies to its head_d dims
        gate_full = gate.repeat_interleave(self.head_d, dim=-1)      # (B, K, dim)
        return gate_full

    # ──────────────────────────────────────────────────────────────────────────
    def forward(self,
                embs:  torch.Tensor,   # (B, K, dim)
                times: torch.Tensor,   # (B, K)
                ) -> torch.Tensor:     # (B, K, dim)
        rope_out   = self._rope_per_head(embs, times)                # (B, K, dim)
        gates      = self._decay_gates(times)                        # (B, K, dim)
        fourier_out = self._fourier(times)                           # (B, K, dim)
        # Each head's RoPE output is gated by its own recency rate,
        # then the absolute Fourier signal is added on top.
        return gates * rope_out + fourier_out                        # (B, K, dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  StructuralTrustModule  — K-VIEW SINKHORN TRAJECTORY SIMILARITY
#
#  Computes trust(i→j) ∈ (0, 2) using ONLY the K-view trajectory signal.
#
#  WHY SINKHORN INSTEAD OF CROSS-ATTENTION?
#  ─────────────────────────────────────────
#  Cross-attention (softmax over rows) lets each of i's views attend to ALL
#  of j's views independently.  Two problems:
#    1. Multiple views of i can "compete" for the same view of j, pulling
#       the match towards a few dominant modes.
#    2. No constraint that the overall matching is globally consistent —
#       a single anomalous view in j can dominate every row.
#
#  Sinkhorn optimal transport solves both:
#    - Finds the globally optimal SOFT ASSIGNMENT matrix T ∈ R^{K×K}
#      between i's K views and j's K views.
#    - Row margins: T·1 = u (how much each view of i is matched)
#    - Column margins: T^T·1 = v (how much each view of j is used)
#    - With uniform margins (u=v=1/K), every view contributes equally.
#    - Minimises sum_{k,k'} C_{k,k'} * T_{k,k'}
#      where C_{k,k'} = 1 - cos(view_i_k, view_j_k') is the COST.
#    - Solved iteratively via Sinkhorn-Knopp (log-domain for stability).
#
#  After finding T, the trajectory similarity is:
#      traj_sim = sum_{k,k'} T_{k,k'} * (1 - C_{k,k'})    (E,) ∈ [-1,1]
#               = sum_{k,k'} T_{k,k'} * cos(vi_k, vj_k')
#  This is the EXPECTED cosine similarity under the optimal transport plan —
#  a true measure of how "alignable" the two trajectories are.
# ═══════════════════════════════════════════════════════════════════════════════

class StructuralTrustModule(nn.Module):
    """
    K-view trajectory trust via Sinkhorn Optimal Transport.

    Args:
        dim         : embedding dimension
        window      : K, number of stored snapshots
        t_heads     : temporal resolution heads for TemporalPositionalEncoding
        sinkhorn_eps: entropy regularisation (smaller = sharper assignment)
        sinkhorn_iters: number of Sinkhorn iterations (5-10 is usually enough)
    """
    def __init__(self, dim: int, window: int = 8,
                 t2v_dim: int = 16,          # kept for API compat, unused
                 t_heads: int = 4,
                 sinkhorn_eps: float = 0.05,
                 sinkhorn_iters: int = 7):
        super().__init__()
        self.window         = window
        self.sinkhorn_eps   = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters

        # Per-head multi-resolution temporal positional encoding
        self.time_pe = TemporalPositionalEncoding(dim, t_heads=t_heads)

        # Lightweight projection to compare views in a lower-dim space
        # This decouples the comparison space from the full hidden dim,
        # reducing computation in the K×K cost matrix.
        proj_dim = max(dim // 2, 32)
        self.proj = nn.Linear(dim, proj_dim, bias=False)

        # Maps OT similarity scalar → trust logit
        self.trust_head = nn.Sequential(
            nn.Linear(1, max(dim // 8, 8)),
            nn.GELU(),
            nn.Linear(max(dim // 8, 8), 1),
        )

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _sinkhorn_log(log_C: torch.Tensor,
                      eps: float, n_iter: int) -> torch.Tensor:
        """
        Log-domain Sinkhorn: finds the optimal transport plan T for cost C.

        log_C : (B, K, K)  log of cost matrix (here log(1 - cos_sim + ε))
        Returns log_T : (B, K, K)  log of transport plan

        Log-domain is numerically stable for small eps.
        The algorithm alternates normalising rows and columns of log_T.
        """
        B, K, _ = log_C.shape
        log_a = torch.full((B, K), -math.log(K),
                           device=log_C.device, dtype=log_C.dtype)  # uniform source
        log_b = torch.full((B, K), -math.log(K),
                           device=log_C.device, dtype=log_C.dtype)  # uniform target

        log_T = -log_C / eps    # initialise with Gibbs kernel
        for _ in range(n_iter):
            # Row normalisation: T·1 = a
            log_T = log_T - torch.logsumexp(log_T, dim=2, keepdim=True) + log_a.unsqueeze(2)
            # Column normalisation: T^T·1 = b
            log_T = log_T - torch.logsumexp(log_T, dim=1, keepdim=True) + log_b.unsqueeze(1)
        return log_T   # (B, K, K)

    # ──────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        embs_i: torch.Tensor,   # (E, K, dim)  last K snapshot embeddings of i
        embs_j: torch.Tensor,   # (E, K, dim)  last K snapshot embeddings of j
        t_i:    torch.Tensor,   # (E, K)       timestamps of i's K snapshots
        t_j:    torch.Tensor,   # (E, K)       timestamps of j's K snapshots
        eps:    float = 1e-6,
    ) -> torch.Tensor:          # (E,) ∈ (0, 2)

        E, K, D = embs_i.shape

        # Step 1: per-head multi-resolution temporal PE
        # Fast heads emphasise recency, slow heads treat all views equally.
        vi = self.time_pe(embs_i, t_i)   # (E, K, dim)
        vj = self.time_pe(embs_j, t_j)

        # Step 2: project to lower-dim comparison space and normalise
        vi_p = F.normalize(self.proj(vi.reshape(E * K, D)),
                           dim=-1, eps=eps).view(E, K, -1)           # (E, K, proj_dim)
        vj_p = F.normalize(self.proj(vj.reshape(E * K, D)),
                           dim=-1, eps=eps).view(E, K, -1)

        # Step 3: full K×K pairwise cosine similarity
        sim = torch.bmm(vi_p, vj_p.transpose(1, 2))                 # (E, K, K)

        # Step 4: Sinkhorn OT — finds globally optimal soft assignment
        log_C = torch.log((1.0 - sim).clamp(min=1e-6))
        log_T = self._sinkhorn_log(log_C, self.sinkhorn_eps,
                                   self.sinkhorn_iters)
        T = log_T.exp()                                              # (E, K, K)

        # Step 5: OT-weighted expected cosine similarity
        traj_sim = (T * sim).sum(dim=(1, 2))                         # (E,) ∈ [-1, 1]

        # Step 6: trust ∈ (0, 2)
        logit = self.trust_head(traj_sim.unsqueeze(-1)).squeeze(-1)
        return 1.0 + torch.tanh(logit)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  MessageTemporalEncoding  — enhanced per-head RoPE + Fourier + decay gate
#
#  Applied to live edge messages (E, dim) before aggregation at node i.
#
#  Four components, all applied together:
#
#  (a) TIMESTAMP NORMALISATION
#      Raw timestamps can be arbitrarily large absolute values (e.g. Unix time).
#      Large t values cause RoPE angles to wrap many times, losing information.
#      We apply learnable normalisation:  t_norm = (t - mu) / (sigma + ε)
#      where mu and sigma are EMA statistics updated during training.
#      This keeps angles in a well-behaved range regardless of the time scale.
#
#  (b) PER-HEAD RoPE
#      Message is split into t_heads segments. Each segment is rotated by
#      RoPE with a head-specific timescale sub-range. Some heads capture
#      fine-grained recency (hourly), others capture long-range patterns.
#
#  (c) MULTI-SCALE FOURIER (additive, shared)
#      Absolute temporal position signal via sin/cos of log-uniform frequencies.
#
#  (d) PER-HEAD TEMPORAL DECAY GATE
#      g_h = σ( -λ_h * |t_norm| + b_h )   ∈ (0, 1)
#      Each head has a learned decay rate λ_h. Fast heads (large λ) strongly
#      down-weight messages that are "old" relative to the mean timestamp.
#      Slow heads weight all messages more uniformly.
#      The gated message for head h: g_h * rope_h(msg_h) + (1-g_h) * msg_h
#      This is a soft interpolation between time-rotated and original content,
#      letting slow heads preserve content while fast heads inject recency.
# ═══════════════════════════════════════════════════════════════════════════════

class MessageTemporalEncoding(nn.Module):
    """
    Per-head multi-resolution RoPE + Fourier + decay gate for edge messages.

    Args:
        dim       : message dimension (even, divisible by t_heads)
        t_heads   : number of temporal resolution heads
        n_fourier : Fourier frequency bands (sin+cos each)
        ema_decay : momentum for running timestamp statistics
    """
    def __init__(self, dim: int, t_heads: int = 4, n_fourier: int = 16,
                 ema_decay: float = 0.99):
        super().__init__()
        assert dim % 2 == 0 and dim % t_heads == 0
        self.dim      = dim
        self.t_heads  = t_heads
        self.head_d   = dim // t_heads
        assert self.head_d % 2 == 0

        # ── (a) Learnable timestamp normalisation ─────────────────────────────
        # Running statistics for EMA normalisation of raw timestamps
        self.ema_decay = ema_decay
        self.register_buffer("t_mean", torch.zeros(1))
        self.register_buffer("t_var",  torch.ones(1))
        # Learnable affine rescale after normalisation
        self.t_scale = nn.Parameter(torch.ones(1))
        self.t_shift = nn.Parameter(torch.zeros(1))

        # ── (b) Per-head RoPE timescales ──────────────────────────────────────
        # Shorter range for messages: angles of normalised timestamps
        rope_params = []
        T_MIN, T_MAX = 0.0, 4.0   # timescales 1 → e^4 ≈ 55 (on normalised scale)
        for h in range(t_heads):
            lo = T_MIN + h       * (T_MAX - T_MIN) / t_heads
            hi = T_MIN + (h + 1) * (T_MAX - T_MIN) / t_heads
            rope_params.append(torch.linspace(lo, hi, self.head_d // 2))
        self.rope_log_ts = nn.Parameter(torch.stack(rope_params, dim=0))  # (H, hd//2)

        # ── (c) Multi-scale Fourier (shared, additive) ────────────────────────
        freqs = torch.logspace(-3, 1, n_fourier)
        self.register_buffer("fourier_freqs", freqs)
        self.fourier_proj = nn.Linear(2 * n_fourier, dim)

        # ── (d) Per-head temporal decay gate ──────────────────────────────────
        # λ_h: learned decay rate per head; fast heads init to larger values
        init_log_decay = torch.linspace(1.5, -0.5, t_heads)  # head 0 = fastest
        self.log_decay  = nn.Parameter(init_log_decay)        # (t_heads,)
        self.decay_bias = nn.Parameter(torch.ones(t_heads))   # (t_heads,)

    @torch.no_grad()
    def _update_stats(self, t: torch.Tensor):
        """EMA update of running mean and variance of timestamps."""
        mu  = t.float().mean()
        var = t.float().var().clamp(min=1e-4)
        self.t_mean = self.ema_decay * self.t_mean + (1 - self.ema_decay) * mu
        self.t_var  = self.ema_decay * self.t_var  + (1 - self.ema_decay) * var

    def _normalise_t(self, t: torch.Tensor) -> torch.Tensor:
        """Normalise timestamps to zero-mean, unit-variance, then rescale."""
        t_norm = (t.float() - self.t_mean) / (self.t_var.sqrt() + 1e-6)
        return self.t_scale * t_norm + self.t_shift      # (E,)

    def forward(self,
                msg: torch.Tensor,   # (E, dim)
                t:   torch.Tensor,   # (E,)   raw edge timestamps
                ) -> torch.Tensor:   # (E, dim)
        E = msg.size(0)

        # Update running stats during training
        if self.training:
            self._update_stats(t)

        # Normalise timestamps
        t_n = self._normalise_t(t)                                   # (E,)

        # ── (b) Per-head RoPE on normalised timestamps ────────────────────────
        rope_parts = []
        for h in range(self.t_heads):
            sl  = msg[:, h * self.head_d : (h + 1) * self.head_d]   # (E, hd)
            ts  = self.rope_log_ts[h].exp()                          # (hd//2,)
            ang = t_n.unsqueeze(-1) / ts                             # (E, hd//2)
            cos_a, sin_a = ang.cos(), ang.sin()
            m_even = sl[:, 0::2]
            m_odd  = sl[:, 1::2]
            rot = torch.stack(
                [m_even * cos_a - m_odd * sin_a,
                 m_even * sin_a + m_odd * cos_a], dim=-1
            ).reshape(E, self.head_d)
            rope_parts.append(rot)
        rope_out = torch.cat(rope_parts, dim=-1)                     # (E, dim)

        # ── (c) Multi-scale Fourier (additive, absolute time) ─────────────────
        phi    = t_n.unsqueeze(-1) * self.fourier_freqs              # (E, F)
        feat   = torch.cat([phi.sin(), phi.cos()], dim=-1)           # (E, 2F)
        fourier = self.fourier_proj(feat)                            # (E, dim)

        # ── (d) Per-head temporal decay gate ──────────────────────────────────
        # g_h = σ( -λ_h * |t_norm| + b_h )
        # Fast heads (large λ) strongly gate old messages toward identity.
        lam  = self.log_decay.exp()                                  # (t_heads,)
        gate = torch.sigmoid(
            -lam.unsqueeze(0) * t_n.abs().unsqueeze(-1)
            + self.decay_bias.unsqueeze(0)
        )                                                            # (E, t_heads)
        # Broadcast gate to full dim (each head covers head_d dims)
        gate_full = gate.repeat_interleave(self.head_d, dim=-1)      # (E, dim)
        # Gated blend: interpolate between rotated and original content
        gated_rope = gate_full * rope_out + (1 - gate_full) * msg    # (E, dim)

        return gated_rope + fourier                                   # (E, dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  GRN  — Gated Residual Network (from Temporal Fusion Transformer, Lim 2021)
#
#  A proven building block for temporal modelling. Replaces plain nn.Linear
#  for Q, K, V projections in TemporalTrustAttention.
#
#  GRN(x, c) = LayerNorm( x + W2 · ELU(W1 · [x, c] + b1) ⊙ σ(W_gate · [x, c]) )
#
#  Two important properties:
#    1. CONTEXT GATING: an optional context vector c (here: the node's GRU
#       memory) modulates the output via a sigmoid gate.  This means the
#       projection adapts to the node's historical state without needing
#       a full hypernetwork.
#    2. ELU NONLINEARITY: unlike GELU (smooth everywhere), ELU has exact
#       linearity for positive inputs and a negative saturation floor.
#       This is beneficial for temporal data where some features are truly
#       linear (trend) and others saturate (periodic amplitude).
#    3. SKIP CONNECTION: the residual allows the GRN to be identity-like
#       when no transformation is needed, which helps early training.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  TemporalMessageBank  — per-node circular buffer with cross-attention
#
#  Stores the last M aggregated messages per node.  After each aggregation,
#  cross-attention reads over this history:
#    Q = current aggregated message   (what I just received)
#    K/V = past M stored messages     (what I have received before)
#  Output: temporally-smoothed message blending current and historical signals.
#
#  Normal nodes: consistent messages → strong history alignment → stable repr.
#  Anomalous influence: divergence from history → dampened across time.
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalMessageBank(nn.Module):
    """
    Per-node circular buffer of past aggregated messages with cross-attention.

    Args:
        num_nodes : total nodes in graph
        dim       : message / hidden dimension
        bank_size : M, number of past messages stored per node
    """
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
        """Cross-attention: current message as Q, historical bank as K/V."""
        past = self.bank[idx]                                        # (N, M, dim)
        Q    = self.q_proj(cur_msg).unsqueeze(1)                     # (N, 1, dim)
        K    = self.k_proj(past)                                     # (N, M, dim)
        V    = self.v_proj(past)
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale         # (N, 1, M)
        attn = F.softmax(attn, dim=-1)
        ctx  = torch.bmm(attn, V).squeeze(1)                        # (N, dim)
        return self.norm(cur_msg + self.out_proj(ctx))               # (N, dim)

    @torch.no_grad()
    def reset(self):
        self.bank.zero_()
        self.ptr.zero_()


class GRN(nn.Module):
    """
    Gated Residual Network projection with optional context conditioning.

    Args:
        in_dim  : input dimension
        out_dim : output dimension
        ctx_dim : context dimension (None = no context gating)
        dropout : dropout before residual add
    """
    def __init__(self, in_dim: int, out_dim: int,
                 ctx_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        ctx = ctx_dim or 0

        # Main transformation: [x, c] -> hidden -> out
        self.W1 = nn.Linear(in_dim + ctx, out_dim)
        self.W2 = nn.Linear(out_dim, out_dim)

        # Context gate: [x, c] -> gate ∈ (0,1) controlling residual weight
        self.W_gate = nn.Linear(in_dim + ctx, out_dim)

        # Skip connection projection (needed when in_dim ≠ out_dim)
        self.skip = (nn.Linear(in_dim, out_dim, bias=False)
                     if in_dim != out_dim else nn.Identity())

        self.norm    = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x:   torch.Tensor,              # (B, in_dim)
                ctx: Optional[torch.Tensor] = None  # (B, ctx_dim)
                ) -> torch.Tensor:              # (B, out_dim)
        if ctx is not None:
            inp = torch.cat([x, ctx], dim=-1)   # (B, in_dim + ctx_dim)
        else:
            inp = x

        # Gated transformation
        h    = F.elu(self.W1(inp))              # (B, out_dim)
        h    = self.dropout(self.W2(h))         # (B, out_dim)
        gate = torch.sigmoid(self.W_gate(inp))  # (B, out_dim) ∈ (0,1)

        # Gated residual: gate blends transformed output with skip
        out = gate * h + (1 - gate) * self.skip(x)
        return self.norm(out)                   # (B, out_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  TemporalTrustAttention  (one message-passing layer)
#
#  [1] GRN-PROJECTED Q / K / V  (context-conditioned via GRU memory)
#      Q  = GRN(edge_emb, ctx=mem_i)  interaction queries through node history
#      K  = GRN(x_j,      ctx=mem_j)  node j's key adapted to its history
#      V  = GRN(x_j,      ctx=mem_j)  value also history-conditioned
#
#  [2] RoPE APPLIED TO Q AND K BEFORE SCORING
#      This is the mathematically correct placement for RoPE.
#      Rotating Q and K by the edge timestamp means that the dot product
#      Q·K naturally captures relative temporal position:
#          (R_t Q)·(R_t K) = Q^T R_{0} K = Q^T K   (same time: unrotated)
#          (R_{t1} Q)·(R_{t2} K) = Q^T R_{t2-t1} K  (different times: rotated)
#      The score between Q and K reflects BOTH semantic compatibility AND
#      temporal proximity. Temporally distant Q-K pairs will have a
#      phase-shifted dot product, naturally reducing their score.
#      Per-head timescales are used (same as MessageTemporalEncoding).
#
#  [3] PER-HEAD BILINEAR SCORING  (replaces identity dot-product)
#      Standard dot-product: score = Q^T K  (uses identity metric)
#      Bilinear form:        score = Q^T W_h K  (each head learns its own metric)
#      W_h ∈ R^{hd×hd} is a learned per-head compatibility matrix.
#      This allows each head to discover a different notion of compatibility
#      between the edge query and the node key — e.g. one head may focus on
#      frequency features, another on structural role features.
#      To keep computation light, W_h is parameterised as a low-rank outer
#      product: W_h = U_h V_h^T where U_h, V_h ∈ R^{hd×r}, rank r << hd.
#
#  [4] SINKHORN TRUST  (K-view OT only, no h/deg)
#
#  [5] PER-HEAD MULTI-RESOLUTION MESSAGE TEMPORAL ENCODING  (with decay gate)
#
#  [6] TEMPORAL MESSAGE BANK
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalTrustAttention(MessagePassing):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1,
                 window: int = 8, t2v_dim: int = 16,
                 num_nodes: int = 0, bank_size: int = 8,
                 t_heads: int = 4, bilinear_rank: int = 8):
        super().__init__(aggr="add", node_dim=0)
        assert dim % heads == 0
        self.dim        = dim
        self.heads      = heads
        self.head_dim   = dim // heads
        self.t_heads    = t_heads

        # ── [1] GRN projections (context-conditioned) ─────────────────────────
        self.q_grn  = GRN(dim, dim, ctx_dim=dim, dropout=dropout)
        self.k_grn  = GRN(dim, dim, ctx_dim=dim, dropout=dropout)
        self.v_grn  = GRN(dim, dim, ctx_dim=dim, dropout=dropout)
        self.o_proj = nn.Linear(dim, dim)

        # ── [2] Per-head RoPE on Q and K ─────────────────────────────────────
        # Timescales shared with message encoding for consistency
        # Each head gets its own sub-range of [0, 4] on normalised timestamps
        rope_params = []
        T_MIN, T_MAX = 0.0, 4.0
        hd = self.head_dim
        assert hd % 2 == 0
        for h in range(heads):
            lo = T_MIN + h       * (T_MAX - T_MIN) / heads
            hi = T_MIN + (h + 1) * (T_MAX - T_MIN) / heads
            rope_params.append(torch.linspace(lo, hi, hd // 2))
        self.qk_rope_log_ts = nn.Parameter(torch.stack(rope_params, dim=0))  # (H, hd//2)

        # Running timestamp stats for normalisation (shared with MessageTemporalEncoding)
        self.register_buffer("t_mean_qk", torch.zeros(1))
        self.register_buffer("t_var_qk",  torch.ones(1))
        self.t_scale_qk = nn.Parameter(torch.ones(1))
        self.t_shift_qk = nn.Parameter(torch.zeros(1))
        self.ema_decay = 0.99

        # ── [3] Per-head bilinear scoring (low-rank) ──────────────────────────
        # W_h = U_h @ V_h^T  where U_h, V_h ∈ R^{hd × r}
        # Parameterised as two (heads, hd, rank) tensors.
        # score_h = Q_h^T W_h K_h = (Q_h^T U_h)(V_h^T K_h) — factored efficiently.
        r = bilinear_rank
        self.bilin_U = nn.Parameter(torch.randn(heads, hd, r) * (r ** -0.5))
        self.bilin_V = nn.Parameter(torch.randn(heads, hd, r) * (r ** -0.5))
        self.scale   = r ** -0.5   # normalise by rank, not head_dim

        # ── [4] Sinkhorn trust (K-view only) ─────────────────────────────────
        self.trust = StructuralTrustModule(dim, window=window, t_heads=t_heads)

        # ── [5] Message temporal encoding (with decay gate) ───────────────────
        self.msg_time_enc = MessageTemporalEncoding(dim, t_heads=t_heads, n_fourier=16)

        # ── [6] Temporal message bank ─────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _update_t_stats(self, t: torch.Tensor):
        mu  = t.float().mean()
        var = t.float().var().clamp(min=1e-4)
        self.t_mean_qk = self.ema_decay * self.t_mean_qk + (1 - self.ema_decay) * mu
        self.t_var_qk  = self.ema_decay * self.t_var_qk  + (1 - self.ema_decay) * var

    def _normalise_t(self, t: torch.Tensor) -> torch.Tensor:
        t_n = (t.float() - self.t_mean_qk) / (self.t_var_qk.sqrt() + 1e-6)
        return self.t_scale_qk * t_n + self.t_shift_qk

    def _apply_rope_to_qk(self,
                           qk:  torch.Tensor,   # (E, heads, head_dim)
                           t_n: torch.Tensor,   # (E,)  normalised timestamps
                           ) -> torch.Tensor:   # (E, heads, head_dim)
        """
        Apply per-head RoPE to Q or K using normalised edge timestamps.
        Each head h rotates its head_dim vector by its own timescale band.
        """
        E = qk.size(0)
        parts = []
        for h in range(self.heads):
            sl  = qk[:, h, :]                                        # (E, hd)
            ts  = self.qk_rope_log_ts[h].exp()                       # (hd//2,)
            ang = t_n.unsqueeze(-1) / ts                             # (E, hd//2)
            cos_a, sin_a = ang.cos(), ang.sin()
            e_even = sl[:, 0::2]
            e_odd  = sl[:, 1::2]
            rot = torch.stack(
                [e_even * cos_a - e_odd * sin_a,
                 e_even * sin_a + e_odd * cos_a], dim=-1
            ).reshape(E, self.head_dim)
            parts.append(rot)
        return torch.stack(parts, dim=1)                             # (E, H, hd)

    # ──────────────────────────────────────────────────────────────────────────
    def forward(self,
                x:          torch.Tensor,   # (N, dim)
                edge_index: torch.Tensor,   # (2, E)
                edge_emb:   torch.Tensor,   # (E, dim)
                time_emb:   torch.Tensor,   # (E, dim) unused, kept for API
                edge_t:     torch.Tensor,   # (E,)  raw timestamps
                memory:     RecurrentMemory,
                evo_bank:   EvolutionBank,
                deg:        torch.Tensor,   # (N,)
                ) -> torch.Tensor:

        N    = x.size(0)
        nidx = torch.arange(N, device=x.device)
        src  = edge_index[0]

        # Update running timestamp statistics
        if self.training:
            self._update_t_stats(edge_t)
        t_n = self._normalise_t(edge_t)                             # (E,)

        mem_vals, _ = memory.read(nidx)
        embs_all, times_all = evo_bank.read(nidx)

        # GRN-projected Q/K/V
        mem_src = mem_vals[src]
        Q = self.q_grn(edge_emb, mem_src).view(-1, self.heads, self.head_dim)
        K = self.k_grn(x, mem_vals).view(N, self.heads, self.head_dim)
        V = self.v_grn(x, mem_vals).view(N, self.heads, self.head_dim)

        # Apply RoPE to Q (edge-level, already (E, H, hd))
        Q_rope = self._apply_rope_to_qk(Q, t_n)                     # (E, H, hd)
        # K will be indexed as K_j inside message() — apply RoPE there

        out = self.propagate(
            edge_index,
            x=x,
            Q_rope=Q_rope,
            K=K,
            V=V,
            mem_val=mem_vals,
            embs=embs_all,
            times=times_all,
            edge_t=edge_t,
            t_n=t_n,
            size=(N, N),
        )
        out = out.view(N, self.dim)

        # Temporal message bank
        if self.msg_bank is not None:
            out = self.msg_bank.read_and_attend(nidx, out)
            self.msg_bank.write(nidx, out)

        # Fuse with GRU memory + gated residual
        h   = memory.get_hidden(nidx)
        out = self.fusion(torch.cat([out, h], dim=-1))
        out = self.o_proj(out)
        g   = torch.sigmoid(self.gate(x))
        return self.norm(g * x + (1 - g) * out)

    # ──────────────────────────────────────────────────────────────────────────
    def message(self,
                x_i:      torch.Tensor,   # (E, dim)
                x_j:      torch.Tensor,
                Q_rope:   torch.Tensor,   # (E, H, hd)  Q already RoPE-encoded
                K_j:      torch.Tensor,   # (E, H, hd)  K from node j (raw)
                V_j:      torch.Tensor,   # (E, H, hd)
                mem_val_i: torch.Tensor,  # (E, dim)
                mem_val_j: torch.Tensor,
                embs_i:   torch.Tensor,   # (E, K, dim)
                embs_j:   torch.Tensor,
                times_i:  torch.Tensor,   # (E, K)
                times_j:  torch.Tensor,
                edge_t:   torch.Tensor,   # (E,)
                t_n:      torch.Tensor,   # (E,)  normalised timestamps
                index:    torch.Tensor,
                ) -> torch.Tensor:

        E = x_i.size(0)

        # ── [2] Apply RoPE to K_j using normalised timestamps ─────────────────
        # After rotation, Q_rope · K_j_rope reflects both semantic match AND
        # temporal proximity between the edge time and j's contribution time.
        K_j_rope = self._apply_rope_to_qk(K_j, t_n)                 # (E, H, hd)

        # ── [3] Per-head bilinear score: score_h = (Q^T U_h)(V_h^T K) ────────
        # Factored bilinear: avoids materialising the full hd×hd matrix.
        # Q^T U_h : (E, r)   and   V_h^T K : (E, r)
        # score_h = dot product of these two r-dim vectors.
        scores = torch.zeros(E, self.heads, device=x_i.device)
        for h in range(self.heads):
            q_h = Q_rope[:, h, :]                                    # (E, hd)
            k_h = K_j_rope[:, h, :]                                  # (E, hd)
            U_h = self.bilin_U[h]                                    # (hd, r)
            V_h = self.bilin_V[h]                                    # (hd, r)
            # (E, r) · (E, r) summed → (E,)
            scores[:, h] = ((q_h @ U_h) * (k_h @ V_h)).sum(-1) * self.scale

        # ── [4] Sinkhorn trust (K-view only) ─────────────────────────────────
        trust = self.trust(
            embs_i=embs_i, embs_j=embs_j,
            t_i=times_i,   t_j=times_j,
        )                                                            # (E,) ∈ (0,2)
        self._trust_cache = trust.detach()

        # Trust-scaled softmax
        scores = scores * trust.unsqueeze(-1)
        scores = softmax(scores, index)
        scores = self.dropout(scores)

        # ── [5] Value + multi-resolution temporal message encoding ────────────
        msg = (V_j * scores.unsqueeze(-1)).reshape(E, self.dim)
        msg = self.msg_time_enc(msg, edge_t)                         # (E, dim)
        return msg


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
        bilinear_rank:   int   = 8,
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
                t_heads=heads, bilinear_rank=bilinear_rank,
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
        # Pass raw edge timestamps (batch.t) separately for RoPE and decay gate
        for layer in self.layers:
            x = layer(x, edge_index, edge_emb, time_emb,
                      batch.t.float(),                               # edge_t
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
        window=K, t2v_dim=16, bank_size=8, bilinear_rank=8,
    )

    for step in range(3):
        emb   = model(make_batch(step * 1000))
        trust = model.get_trust_scores()
        print(f"[pass {step+1}] emb: {emb.shape} | "
              f"trust: [{trust.min():.4f}, {trust.max():.4f}] "
              f"mean={trust.mean():.4f}")
        assert trust.min() > 0.0 and trust.max() < 2.0

    print("\nAll checks passed. Trust is in (0, 2).")