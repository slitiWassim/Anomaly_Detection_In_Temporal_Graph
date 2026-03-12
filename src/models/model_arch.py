"""
Temporal Trust GNN for Unsupervised Anomaly Detection  (v2)
============================================================
Produces node representations for downstream contrastive learning.
No labels required — fully unsupervised.

Message Aggregation (TemporalTrustAttention) — UPDATED in v2:

  [1] REVISED CROSS-SOURCE ATTENTION
      Q comes from SOURCE node features  x_src  (what am I looking for?)
      K comes from TARGET node features  CONCAT  edge features
                                                  (who are you, via this edge?)
      V comes from TARGET node features  CONCAT  edge features
                                                  (what do you carry, in context?)

      This inverts the previous edge-Q / node-K design so that the SOURCE node
      drives the query and the edge context shapes both key and value.

  [2] RoPE ON Q, K AND V  (MessageRoPE)
      All three projections are rotated by the edge timestamp BEFORE the dot
      product is computed.  The attention score therefore captures:
        • structural relevance  (content alignment after rotation)
        • temporal alignment   (same-time pairs stay aligned; cross-time pairs
                                rotate apart)
      V is also rotated so the aggregated sum carries intrinsic time information.

  [3] LeakyReLU-GATED SOFTMAX
      Raw scores are passed through LeakyReLU before softmax:
          α = softmax( LeakyReLU( Q·K / sqrt(d) ) )
      Negative scores receive a small gradient instead of being hard-zeroed,
      which prevents the "dead attention head" problem and lets the model learn
      to actively suppress irrelevant neighbours rather than merely ignoring them.

  [4] SIMILARITY-BASED TRUST  (StructuralTrustModule, unchanged)
      trust(i->j) in (0, 2) — amplifies similar pairs, attenuates dissimilar.

  [5] TEMPORAL MESSAGE BANK  (TemporalMessageBank, unchanged)
      Per-node circular buffer of past aggregated messages; cross-attention
      blends current with historical received signals.
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
# 6.  TemporalPositionalEncoding  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalPositionalEncoding(nn.Module):
    """RoPE + Time2Vec positional encoding for K snapshot embeddings."""
    def __init__(self, dim: int, t2v_dim: int = 16):
        super().__init__()
        assert dim % 2 == 0
        self.dim     = dim
        self.t2v_dim = t2v_dim
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
        B, K, D = embs.shape
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
# 8.  MessageRoPE  — continuous-time RoPE for edge-level vectors  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class MessageRoPE(nn.Module):
    """
    Rotates a batch of vectors (E, dim) by angles proportional to edge
    timestamps.  Used independently for Q, K, and V in v2.
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.log_timescales = nn.Parameter(torch.linspace(0, 6, dim // 2))

    def forward(self, vec: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """vec: (E, dim),  t: (E,)  ->  (E, dim)"""
        timescales = self.log_timescales.exp()
        angles     = t.float().unsqueeze(-1) / timescales       # (E, D//2)
        cos_a, sin_a = angles.cos(), angles.sin()

        v_even = vec[:, 0::2]
        v_odd  = vec[:, 1::2]
        rot_even = v_even * cos_a - v_odd * sin_a
        rot_odd  = v_even * sin_a + v_odd * cos_a

        return torch.stack([rot_even, rot_odd], dim=-1).reshape(vec.shape)


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  TemporalMessageBank  (unchanged)
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
        Q   = self.q_proj(cur_msg).unsqueeze(1)                  # (N, 1, dim)
        K   = self.k_proj(past)                                  # (N, M, dim)
        V   = self.v_proj(past)
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale     # (N, 1, M)
        attn = F.softmax(attn, dim=-1)
        ctx  = torch.bmm(attn, V).squeeze(1)                    # (N, dim)
        return self.norm(cur_msg + self.out_proj(ctx))

    @torch.no_grad()
    def reset(self):
        self.bank.zero_()
        self.ptr.zero_()


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  TemporalTrustAttention  — REVISED attention mechanism
#
#  WHAT CHANGED vs v1
#  ──────────────────
#  [1] Q  ←  source node features  x_src  (was: edge embedding)
#      K  ←  concat(target node features, edge features) → Linear(2d, d)
#      V  ←  concat(target node features, edge features) → Linear(2d, d)
#
#      Motivation: the SOURCE node asks "what am I looking for?";
#      the TARGET node answers "here is who I am, given this specific edge."
#      Edge context is fused into K and V so the same target node can
#      present a different face on each incoming edge type/context.
#
#  [2] RoPE applied to Q, K AND V individually
#      Each has its own MessageRoPE with independently learned timescales.
#      • Rotating Q and K makes the attention score sensitive to the TEMPORAL
#        DISTANCE between source and target at the moment of the interaction.
#      • Rotating V ensures the aggregated message at the target carries
#        intrinsic time information (not just score-based recency weighting).
#
#  [3] LeakyReLU-gated softmax:
#          α = softmax( LeakyReLU( Q·K / sqrt(d),  slope=0.2 ) )
#      Negative raw scores receive a small non-zero gradient, preventing
#      "dead attention heads" and giving the model an explicit mechanism
#      to SUPPRESS certain neighbours (negative → near-zero after softmax)
#      while still back-propagating a learning signal through them.
#
#  Everything else (trust, message bank, GRU fusion) is unchanged.
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalTrustAttention(MessagePassing):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1,
                 window: int = 8, t2v_dim: int = 16,
                 num_nodes: int = 0, bank_size: int = 8,
                 leaky_slope: float = 0.2):
        super().__init__(aggr="add", node_dim=0)
        assert dim % heads == 0
        self.dim         = dim
        self.heads       = heads
        self.head_dim    = dim // heads
        self.leaky_slope = leaky_slope

        # ── [NEW] Projections ─────────────────────────────────────────────────
        # Q: source node features  (dim -> dim)
        self.q_src_proj = nn.Linear(dim, dim)

        # K: target node features + edge features  (2*dim -> dim)
        self.k_tgt_proj = nn.Linear(dim * 2, dim)

        # V: target node features + edge features  (2*dim -> dim)
        self.v_tgt_proj = nn.Linear(dim * 2, dim)

        self.o_proj = nn.Linear(dim, dim)
        self.scale  = self.head_dim ** -0.5

        # ── [NEW] Separate RoPE modules for Q, K, V ───────────────────────────
        # Each learns its own timescales for the appropriate regime:
        #   q_rope — source query: coarser timescales (node-level activity)
        #   k_rope — target+edge key: medium timescales
        #   v_rope — target+edge value: finer timescales (message content)
        self.q_rope = MessageRoPE(dim)
        self.k_rope = MessageRoPE(dim)
        self.v_rope = MessageRoPE(dim)

        # ── Trust ─────────────────────────────────────────────────────────────
        self.trust = StructuralTrustModule(dim, window=window, t2v_dim=t2v_dim)

        # ── Temporal message bank ─────────────────────────────────────────────
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

        N    = x.size(0)
        E    = edge_index.size(1)
        src, dst = edge_index                                        # (E,), (E,)
        nidx = torch.arange(N, device=x.device)
        t    = edge_t.float()

        # ── [1] Build Q from SOURCE node features ─────────────────────────────
        # Q shape: (E, dim) — one query vector per edge, drawn from the source
        Q = self.q_src_proj(x[src])                                # (E, dim)
        # Apply RoPE: rotate query by edge timestamp
        Q = self.q_rope(Q, t)                                        # (E, dim)
        Q = Q.view(E, self.heads, self.head_dim)                     # (E, H, hd)

        # ── [2] Build K from TARGET node + edge features ──────────────────────
        # Concatenate target node representation with edge embedding,
        # then project down to dim.  This lets the key encode BOTH who the
        # target node is AND the relational context of this specific edge.
        K = self.k_tgt_proj(
                torch.cat([x[dst], edge_emb], dim=-1))               # (E, dim)
        K = self.k_rope(K, t)                                        # (E, dim)
        K = K.view(E, self.heads, self.head_dim)                     # (E, H, hd)

        # ── [3] Build V from TARGET node + edge features ──────────────────────
        # Same fusion as K, but with a separate learned projection so that
        # "what to send" can differ from "how to be indexed".
        V = self.v_tgt_proj(
                torch.cat([x[dst], edge_emb], dim=-1))               # (E, dim)
        V = self.v_rope(V, t)                                        # (E, dim)
        V = V.view(E, self.heads, self.head_dim)                     # (E, H, hd)

        # Pre-fetch memory and evolution bank for all nodes
        mem_vals, _ = memory.read(nidx)
        embs_all, times_all = evo_bank.read(nidx)

        # ── Propagate ─────────────────────────────────────────────────────────
        # Pass pre-computed (E, H, hd) tensors directly — no _i/_j indexing
        # needed inside message() since they are already edge-level.
        out = self.propagate(
            edge_index,
            x        = x,
            Q        = Q,            # (E, H, hd)  — edge-level, pass as-is
            K        = K,            # (E, H, hd)
            V        = V,            # (E, H, hd)
            mem_val  = mem_vals,     # (N, dim)    — will become mem_val_i/j
            deg      = deg,          # (N,)        — will become deg_i/j
            embs     = embs_all,     # (N, K, dim) — will become embs_i/j
            times    = times_all,    # (N, K)      — will become times_i/j
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
                x_i:       torch.Tensor,   # (E, dim)  target node features
                x_j:       torch.Tensor,   # (E, dim)  source node features
                Q:         torch.Tensor,   # (E, H, hd) pre-computed, edge-level
                K:         torch.Tensor,   # (E, H, hd)
                V:         torch.Tensor,   # (E, H, hd)
                mem_val_i: torch.Tensor,   # (E, dim)
                mem_val_j: torch.Tensor,
                deg_i:     torch.Tensor,   # (E,)
                deg_j:     torch.Tensor,
                embs_i:    torch.Tensor,   # (E, K, dim)
                embs_j:    torch.Tensor,
                times_i:   torch.Tensor,   # (E, K)
                times_j:   torch.Tensor,
                index:     torch.Tensor,   # (E,)  target indices for softmax
                ) -> torch.Tensor:         # (E, dim)

        E = x_i.size(0)

        # ── [1] Scaled dot-product attention (Q from src, K from tgt+edge) ────
        # Q and K are already per-edge and RoPE-encoded.
        # Dot product over head dimension: (E, H, hd) -> (E, H)
        raw_scores = (Q * K).sum(-1) * self.scale                   # (E, H)

        # ── [2] Similarity-based trust ────────────────────────────────────────
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

        # Modulate raw scores by structural trust BEFORE activation.
        # Similar pairs (trust > 1) get amplified; dissimilar (trust < 1) get
        # attenuated.  Applied per-head so each head can specialise.
        raw_scores = raw_scores * trust.unsqueeze(-1)                # (E, H)

        # ── [3] LeakyReLU-gated softmax ───────────────────────────────────────
        # LeakyReLU before softmax:
        #   • positive scores: pass through unchanged (standard competition)
        #   • negative scores: leak a fraction (slope=leaky_slope) instead of
        #     zeroing — model can learn to SUPPRESS rather than merely ignore
        #   • softmax normalises across all neighbours of the same target node
        activated_scores = F.leaky_relu(raw_scores,
                                        negative_slope=self.leaky_slope)  # (E, H)
        
        attn = softmax(activated_scores, index)                     # (E, H)
        attn = self.dropout(attn)                                   # (E, H)

        # ── [4] Weighted sum over V (already RoPE-encoded in forward) ─────────
        # V carries time-rotated target+edge content.
        # Weight each head's value slice by its attention score.
        msg = (V * attn.unsqueeze(-1))                              # (E, H, hd)
        return msg.reshape(E, self.dim)                             # (E, dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  EnhancedTemporalGNN  — full model  (unchanged except layer instantiation)
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
        leaky_slope:     float = 0.2,
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
        window=K, t2v_dim=16, bank_size=8, leaky_slope=0.2,
    )

    for step in range(3):
        emb   = model(make_batch(step * 1000))
        trust = model.get_trust_scores()
        print(f"[pass {step+1}] emb: {emb.shape} | "
              f"trust: [{trust.min():.4f}, {trust.max():.4f}] "
              f"mean={trust.mean():.4f}")
        assert trust.min() > 0.0 and trust.max() < 2.0

    print("\nAll checks passed.  Trust is in (0, 2).")