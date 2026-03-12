"""
Neighborhood-Aware Temporal Graph Attention Network  (NATGAT)
=============================================================
Representation-learning encoder only.

Returns a single unified node embedding  z ∈ ℝ^hidden_dim  that is
jointly aware of trust-filtered neighborhood structure and the node's
own temporal evolution — feed  z  directly into your contrastive objective.

Design goals
------------
1. TEMPORAL MODELING via K-step recurrent view buffer
   Each node keeps a rolling window of its K most recent embedding
   snapshots together with their timestamps.  DeltaTimeRNN processes
   the full K-step sequence through a GRU whose input gate is explicitly
   modulated by the inter-view time gap Δt, so the model natively
   captures high-frequency bursts (small Δt) and periodic rhythms
   (recurring Δt patterns).  FreqAwareEmbed encodes Δt with a learnable
   log-spaced sinusoidal bank (both cos and sin) that can adapt to the
   dataset's actual temporal scales.

2. TRUST-GATED AGGREGATION via three-signal TrustScorer
   For each directed edge i←j a scalar trust τ(i,j) ∈ (0,1) is computed
   from:
     (A) Attribute alignment  — cosine similarity of current embeddings
     (B) Activity divergence  — cross-attention between i's and j's
                                 K-step hidden-state sequences
     (C) Edge plausibility    — MLP over the edge attribute embedding
   Trust is applied BEFORE softmax so suspicious neighbors are
   down-weighted both in their share of the attention distribution
   and in the information they carry.

3. EDGE ATTRIBUTES as first-class signals
   Edge features enrich both attention keys (routing) and values
   (content), ensuring edge-level anomalies affect both where the
   model looks and what it receives.

Components
----------
FreqAwareEmbed   — Δt → learnable sinusoidal embedding
ViewBuffer       — rolling K-snapshot circular buffer per node
DeltaTimeRNN     — Δt-gated GRU over K views → temporal summary
TrustScorer      — per-edge trust from attribute + activity + edge
NATGATConv       — trust-gated multi-head message-passing layer
NATGAT           — full encoder; returns unified z  (N, hidden_dim)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  FreqAwareEmbed  —  scalar Δt → learnable multi-frequency embedding
# ═══════════════════════════════════════════════════════════════════════════════

class FreqAwareEmbed(nn.Module):
    """
    Δt  →  [linear,  cos(w₁Δt+φ₁), sin(w₁Δt+φ₁), …,
                     cos(wₙΔt+φₙ), sin(wₙΔt+φₙ)]  →  MLP  →  ℝᵈ

    Frequencies w are initialised on a log-scale spanning five decades
    (0.01 … 100 rad/unit) and are *learned*, so the model discovers the
    dataset's characteristic temporal scales automatically.
    Both cos and sin are used so any phase can be represented without
    cancellation artefacts.
    """

    def __init__(self, out_dim: int, num_freqs: int = None):
        super().__init__()
        if num_freqs is None:
            num_freqs = (out_dim - 1) // 2          # 1 linear + 2·F sinusoidal

        # Base log-spaced frequencies (fixed initialisation, then learned)
        base = torch.logspace(-2, 2, num_freqs)     # (F,)
        self.register_buffer("base_freq", base)
        self.log_freq_scale = nn.Parameter(torch.zeros(num_freqs))  # additive log-scale
        self.phase          = nn.Parameter(torch.zeros(num_freqs))  # phase offset

        raw_dim = 1 + 2 * num_freqs
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.num_freqs = num_freqs
        self.out_dim   = out_dim

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """dt : (...,)  →  (..., out_dim)"""
        shape   = dt.shape
        dt_flat = dt.float().reshape(-1, 1)                      # (N, 1)

        freqs  = self.base_freq * torch.exp(self.log_freq_scale) # (F,)
        angles = dt_flat * freqs + self.phase                    # (N, F)
        raw    = torch.cat([dt_flat,
                            torch.cos(angles),
                            torch.sin(angles)], dim=-1)          # (N, 1+2F)
        return self.proj(raw).reshape(*shape, self.out_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ViewBuffer  —  rolling K-snapshot circular buffer per node
# ═══════════════════════════════════════════════════════════════════════════════

class ViewBuffer(nn.Module):
    """
    Stores the K most recent (embedding, timestamp) snapshots for every node.
    Snapshots are written after each forward pass; reads return them ordered
    oldest → newest together with inter-view time deltas Δt.

    Buffers
    -------
    emb  : (num_nodes, K, dim)
    time : (num_nodes, K)
    ptr  : (num_nodes,)   circular write pointer
    """

    def __init__(self, num_nodes: int, dim: int, K: int = 8):
        super().__init__()
        self.K   = K
        self.dim = dim
        self.register_buffer("emb",  torch.zeros(num_nodes, K, dim))
        self.register_buffer("time", torch.zeros(num_nodes, K))
        self.register_buffer("ptr",  torch.zeros(num_nodes, dtype=torch.long))

    def read(self, idx: torch.Tensor):
        """
        idx : (B,)
        Returns
        -------
        views : (B, K, dim)   oldest-first
        times : (B, K)        absolute timestamps
        dts   : (B, K)        Δt[k] = time[k] − time[k−1], 0 at k=0
        """
        B     = idx.size(0)
        views = self.emb[idx]   # (B, K, D)
        times = self.time[idx]  # (B, K)

        # Reorder so slot 0 = oldest
        rolls = (-self.ptr[idx]) % self.K
        views = torch.stack([torch.roll(views[b], int(rolls[b]), dims=0)
                             for b in range(B)])
        times = torch.stack([torch.roll(times[b], int(rolls[b]), dims=0)
                             for b in range(B)])

        dts = torch.zeros_like(times)
        dts[:, 1:] = (times[:, 1:] - times[:, :-1]).clamp(min=0)
        return views, times, dts

    @torch.no_grad()
    def update(self, idx: torch.Tensor, emb: torch.Tensor, t: torch.Tensor):
        """
        idx : (B,)  |  emb : (B, dim)  |  t : (B,)
        Write new snapshot into the circular buffer.
        """
        for b in range(idx.size(0)):
            p = int(self.ptr[idx[b]]) % self.K
            self.emb[idx[b],  p] = emb[b].detach()
            self.time[idx[b], p] = t[b].detach()
            self.ptr[idx[b]]    += 1


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  DeltaTimeRNN  —  Δt-gated GRU over K historical views
# ═══════════════════════════════════════════════════════════════════════════════

class DeltaTimeRNN(nn.Module):
    """
    Processes a node's K-step view sequence through a GRU where the input
    is explicitly modulated by the inter-view time gap Δt.

    Intuition
    ---------
    Large Δt  → node may have drifted; gate opens wide to absorb the new snapshot.
    Small Δt  → burst activity; gate is more conservative (consecutive observations
                are correlated, so marginal information is lower).

    The gate modulation is *learned*: Δt is encoded by FreqAwareEmbed and
    projected to a multiplicative mask over the snapshot embedding before the GRU
    sees it.

    Outputs
    -------
    h_final : (B, dim)    — temporal summary of the node's trajectory
    h_seq   : (B, K, dim) — all GRU hidden states (used by TrustScorer)
    """

    def __init__(self, dim: int, K: int = 8, num_freqs: int = None):
        super().__init__()
        self.dim = dim
        self.K   = K

        self.dt_embed    = FreqAwareEmbed(dim, num_freqs=num_freqs)
        self.dt_gate_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        # Input = gated snapshot ‖ Δt embedding → 2·dim
        self.gru      = nn.GRU(input_size=dim * 2, hidden_size=dim, batch_first=True)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, views: torch.Tensor, dts: torch.Tensor):
        """
        views : (B, K, dim)
        dts   : (B, K)        inter-view Δt (0 at position 0)
        """
        dt_emb  = self.dt_embed(dts)             # (B, K, dim)
        dt_gate = self.dt_gate_proj(dt_emb)      # (B, K, dim) ∈ (0,1)

        inp      = torch.cat([views * dt_gate, dt_emb], dim=-1)  # (B, K, 2·dim)
        h_seq, _ = self.gru(inp)                                  # (B, K, dim)
        h_final  = self.out_norm(h_seq[:, -1, :])                 # (B, dim)
        return h_final, h_seq


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TrustScorer  —  per-edge trust from attribute + activity + edge signal
# ═══════════════════════════════════════════════════════════════════════════════

class TrustScorer(nn.Module):
    """
    Computes τ(i→j) ∈ (0,1) for each directed edge from three signals:

    (A) Attribute alignment
        cosine_similarity(xᵢ, xⱼ) mapped to (0,1).
        Structurally similar nodes should trust each other more.

    (B) Activity alignment
        Cross-attention diagonal of i's GRU sequence vs j's GRU sequence.
        High diagonal alignment ≈ compatible temporal histories.
        Anomalous nodes often have alien behavioral patterns.

    (C) Edge plausibility
        MLP(edge_emb) → (0,1).
        Unusual edge types / magnitudes reduce trust independently of node
        similarity.

    A small MLP combines the three scalars into the final trust score.
    """

    def __init__(self, dim: int, K: int = 8, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # (B) cross-attention projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)

        # (C) edge plausibility
        self.edge_plaus = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

        # Final combination
        self.out_proj = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self,
                x_i:      torch.Tensor,   # (E, dim)
                x_j:      torch.Tensor,   # (E, dim)
                h_seq_i:  torch.Tensor,   # (E, K, dim)
                h_seq_j:  torch.Tensor,   # (E, K, dim)
                edge_emb: torch.Tensor,   # (E, dim)
                ) -> torch.Tensor:        # (E,) ∈ (0,1)

        # ── (A) Attribute alignment ──────────────────────────────────────────
        attr_score = (F.cosine_similarity(x_i, x_j, dim=-1) + 1.0) / 2.0   # (E,)

        # ── (B) Activity alignment via cross-attention diagonal ──────────────
        E, K, D = h_seq_i.shape
        nh, hd  = self.num_heads, self.head_dim

        Q  = self.q_proj(h_seq_i.reshape(E * K, D)).view(E, K, nh, hd)
        Kv = self.k_proj(h_seq_j.reshape(E * K, D)).view(E, K, nh, hd)

        Q_t  = Q.permute(0, 2, 1, 3)    # (E, nh, K, hd)
        Kv_t = Kv.permute(0, 2, 3, 1)   # (E, nh, hd, K)
        attn = F.softmax(
            torch.matmul(Q_t, Kv_t) * self.scale, dim=-1)         # (E, nh, K, K)

        # Mean diagonal = fraction of attention that stays "on-step" → alignment
        activity_score = torch.diagonal(attn, dim1=-2, dim2=-1).mean(dim=(-1, -2))  # (E,)

        # ── (C) Edge plausibility ─────────────────────────────────────────────
        edge_score = self.edge_plaus(edge_emb).squeeze(-1)         # (E,)

        # ── Combine ───────────────────────────────────────────────────────────
        signals = torch.stack([attr_score, activity_score, edge_score], dim=-1)  # (E,3)
        return torch.sigmoid(self.out_proj(signals).squeeze(-1))                 # (E,)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  NATGATConv  —  trust-gated message-passing layer
# ═══════════════════════════════════════════════════════════════════════════════

class NATGATConv(MessagePassing):
    """
    Single trust-gated attention layer.

    For target node i aggregating from neighbors j:

        msg(j→i) = (V_j + ev_j) · τ(i,j) · softmax_α(i,j)

    Trust τ multiplies attention BEFORE softmax so anomalous neighbors are
    reduced both in their attention weight and relative share.

    Edge attributes contribute to:
      - Keys  : via ek_j → routing (which neighbors are attended to)
      - Values: via ev_j → content (what information is carried)
    """

    def __init__(self, dim: int, heads: int = 8, K: int = 8,
                 dropout: float = 0.1, num_freqs: int = None):
        super().__init__(aggr="add", node_dim=0)
        assert dim % heads == 0
        self.dim      = dim
        self.heads    = heads
        self.head_dim = dim // heads
        self.scale    = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(dim, dim)
        self.k_proj   = nn.Linear(dim, dim)
        self.v_proj   = nn.Linear(dim, dim)
        self.o_proj   = nn.Linear(dim, dim)

        self.edge_k_proj = nn.Linear(dim, dim)   # edge → key bias
        self.edge_v_proj = nn.Linear(dim, dim)   # edge → value bias
        self.time_k_proj = nn.Linear(dim, dim)   # time enc → key bias

        self.trust_scorer = TrustScorer(dim, K=K, num_heads=max(1, heads // 2))

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(dim)

        # Simple residual after aggregation — temporal fusion is done once at
        # the model level after all layers, not per-layer
        self.gate = nn.Linear(dim, 1)

        self._trust_cache: torch.Tensor = None

    def split(self, x):
        return x.view(-1, self.heads, self.head_dim)

    def forward(self, x, edge_index, edge_emb, time_emb, h_seq):
        """
        x          : (N, dim)
        edge_index : (2, E)
        edge_emb   : (E, dim)   edge attribute embedding
        time_emb   : (E, dim)   timestamp encoding
        h_seq      : (N, K, dim) DeltaTimeRNN sequence (for TrustScorer)
        """
        N   = x.size(0)
        src = edge_index[0]
        dst = edge_index[1]

        Q      = self.split(self.q_proj(x))       # (N, heads, hd)
        K_base = self.split(self.k_proj(x))
        V_base = self.split(self.v_proj(x))

        ek = self.split(self.edge_k_proj(edge_emb))   # (E, heads, hd)
        ev = self.split(self.edge_v_proj(edge_emb))
        tk = self.split(self.time_k_proj(time_emb))

        # Per-edge trust — computed from node attributes, K-step activity
        # histories, and edge features; modulates aggregation before softmax
        trust = self.trust_scorer(
            x_i      = x[src],
            x_j      = x[dst],
            h_seq_i  = h_seq[src],
            h_seq_j  = h_seq[dst],
            edge_emb = edge_emb,
        )                                              # (E,)
        self._trust_cache = trust.detach()

        out = self.propagate(
            edge_index,
            Q=Q, K_base=K_base, V_base=V_base,
            ek=ek, ev=ev, tk=tk,
            trust=trust,
            size=(N, N),
        )                                              # (N, heads, hd)

        out = self.o_proj(out.view(N, self.dim))

        # Gated residual — keeps the node's own signal stable
        g = torch.sigmoid(self.gate(x))
        return self.norm(g * x + (1 - g) * out)

    def message(self, Q_i, K_base_j, V_base_j,
                ek, ev, tk, trust, index):
        # Enrich keys with edge type and temporal bias
        K_j = K_base_j + ek + tk                          # (E, heads, hd)
        # Enrich values with edge content signal
        V_j = V_base_j + ev                               # (E, heads, hd)

        # Attention score, trust-gated before softmax
        attn = (Q_i * K_j).sum(-1) * self.scale           # (E, heads)
        attn = attn * trust.unsqueeze(-1)                  # trust modulation
        attn = softmax(attn, index)                        # (E, heads)
        attn = self.dropout(attn)

        return V_j * attn.unsqueeze(-1)                    # (E, heads, hd)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  NATGAT  —  full representation-learning encoder
# ═══════════════════════════════════════════════════════════════════════════════

class NATGAT(nn.Module):
    """
    Neighborhood-Aware Temporal Graph Attention Network — representation encoder.

    Produces a single unified node embedding  z ∈ ℝ^hidden_dim  that jointly encodes:

      • Trust-filtered neighborhood structure  (trust-gated GNN layers)
        Per-edge trust is computed from attribute alignment, K-step activity
        history divergence, and edge plausibility — and multiplies attention
        weights BEFORE softmax so anomalous neighbors are suppressed in both
        routing and message content.

      • Node temporal evolution  (DeltaTimeRNN over K historical views)
        The node's K most recent snapshots are processed with explicit Δt
        gating so the model distinguishes bursts, drifts and periodic patterns.

    Both signals are fused at the final readout via a learned cross-gate.
    Feed  z  directly into your contrastive / self-supervised objective.

    Parameters
    ----------
    num_nodes  : total nodes in the graph
    in_dim     : raw node feature size
    edge_dim   : edge attribute size
    hidden_dim : internal embedding size  (must be divisible by heads)
    num_layers : number of NATGATConv layers
    heads      : attention heads per layer
    K          : history window — number of past views kept per node
    dropout    : attention dropout
    num_freqs  : sinusoidal frequency count (None = auto from hidden_dim)
    """

    def __init__(
        self,
        num_nodes:  int,
        in_dim:     int,
        edge_dim:   int,
        hidden_dim: int,
        num_layers: int   = 2,
        heads:      int   = 8,
        K:          int   = 8,
        dropout:    float = 0.1,
        num_freqs:  int   = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes  = num_nodes
        self.K          = K

        # Input projections
        self.node_proj = nn.Linear(in_dim,   hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # Timestamp encoding (absolute + relative)
        self.abs_t_embed = FreqAwareEmbed(hidden_dim // 2, num_freqs=num_freqs)
        self.rel_t_embed = FreqAwareEmbed(hidden_dim // 2, num_freqs=num_freqs)
        self.time_proj   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Per-node K-step history
        self.view_buffer = ViewBuffer(num_nodes, hidden_dim, K=K)

        # Recurrent temporal summarizer
        self.delta_rnn = DeltaTimeRNN(hidden_dim, K=K, num_freqs=num_freqs)

        # Message-passing stack
        self.layers = nn.ModuleList([
            NATGATConv(dim=hidden_dim, heads=heads, K=K,
                       dropout=dropout, num_freqs=num_freqs)
            for _ in range(num_layers)
        ])

        # ── Final readout: fuse structural (GNN) + temporal (DeltaTimeRNN) views
        # into a single unified representation that is simultaneously aware of:
        #   • trust-filtered neighborhood structure  (from the GNN layers)
        #   • the node's own temporal evolution      (from DeltaTimeRNN)
        # A cross-gating mechanism lets the model decide per-node and per-dim
        # how much to rely on each signal.
        self.readout_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.readout_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def _encode_time(self, t_abs, t_rel):
        """Combine absolute + relative timestamp encodings → (E, hidden_dim)"""
        return self.time_proj(
            torch.cat([self.abs_t_embed(t_abs),
                       self.rel_t_embed(t_rel)], dim=-1))

    # ------------------------------------------------------------------
    def get_temporal_repr(self, node_ids: torch.Tensor):
        """
        Read K-step history from ViewBuffer and run DeltaTimeRNN.

        Returns
        -------
        h_temporal : (B, hidden_dim)   final GRU state (temporal view)
        h_seq      : (B, K, hidden_dim) full GRU sequence (for TrustScorer)
        """
        views, _, dts      = self.view_buffer.read(node_ids)
        h_temporal, h_seq  = self.delta_rnn(views, dts)
        return h_temporal, h_seq

    # ------------------------------------------------------------------
    def forward(self, batch):
        """
        batch attributes required
        -------------------------
        x          (N, in_dim)    node features
        edge_index (2, E)
        msg        (E, edge_dim)  edge attributes / message features
        t          (E,)           edge timestamps

        Returns
        -------
        z : (N, hidden_dim)
            Unified node embedding that jointly encodes:
              • trust-filtered neighborhood structure  (trust-gated GNN layers)
              • the node's own temporal trajectory     (DeltaTimeRNN over K views)
            Feed directly into your contrastive / self-supervised objective.
        """
        device   = batch.x.device
        N        = batch.x.size(0)
        src      = batch.edge_index[0]
        all_ids  = torch.arange(N, device=device)

        # ── Project inputs ────────────────────────────────────────────────────
        x        = self.node_proj(batch.x)          # (N, D)
        edge_emb = self.edge_proj(batch.msg)         # (E, D)

        # ── Temporal encoding per edge (absolute + relative to node's last view)
        _, times_all, _ = self.view_buffer.read(all_ids)
        node_last_t     = times_all[:, -1]           # (N,) most recent snapshot time

        t_abs    = batch.t.float()
        t_rel    = (t_abs - node_last_t[src]).clamp(min=0.0)    # (E,)
        time_emb = self._encode_time(t_abs, t_rel)              # (E, D)

        # ── Temporal summaries from K-step history ────────────────────────────
        # h_temporal : node's own trajectory summary  (N, D)
        # h_seq      : full K-step sequence           (N, K, D)  → TrustScorer
        h_temporal, h_seq = self.get_temporal_repr(all_ids)

        # ── Trust-gated message passing ───────────────────────────────────────
        # Each layer uses h_seq to compute per-edge trust, so suspicious
        # neighbors are down-weighted based on activity divergence, attribute
        # misalignment and edge plausibility — all before softmax normalisation.
        for layer in self.layers:
            x = layer(
                x          = x,
                edge_index  = batch.edge_index,
                edge_emb    = edge_emb,
                time_emb    = time_emb,
                h_seq       = h_seq,
            )
        # x here is the trust-filtered structural view  (N, D)

        # ── Final readout: fuse structural + temporal into one representation ─
        # Cross-gate learns per-node, per-dim how much to rely on each signal:
        #   • x          carries neighborhood context filtered by trust
        #   • h_temporal carries the node's own behavioral evolution over time
        # Together they make the representation aware of BOTH suspicious
        # neighbors AND temporal dynamics without leaking either as a
        # separate output to the downstream objective.
        concat = torch.cat([x, h_temporal], dim=-1)              # (N, 2D)
        gate   = self.readout_gate(concat)                       # (N, D) ∈ (0,1)
        proj   = self.readout_proj(concat)                       # (N, D)
        z      = self.final_norm(gate * x + (1 - gate) * proj)   # (N, D)

        # ── Update ViewBuffer with the unified embedding for the next step ────
        node_cur_t = torch.zeros(N, device=device)
        node_cur_t.scatter_reduce_(0, src, t_abs, reduce='amax', include_self=True)
        self.view_buffer.update(all_ids, z, node_cur_t)

        return z


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import types, random

    torch.manual_seed(42)
    random.seed(42)

    N, E      = 64, 256
    in_dim    = 32
    edge_dim  = 16
    hidden    = 64
    K         = 8
    num_nodes = 128

    batch = types.SimpleNamespace()
    batch.x          = torch.randn(N, in_dim)
    batch.edge_index = torch.randint(0, N, (2, E))
    batch.msg        = torch.randn(E, edge_dim)
    batch.t          = torch.rand(E) * 1000

    model = NATGAT(
        num_nodes  = num_nodes,
        in_dim     = in_dim,
        edge_dim   = edge_dim,
        hidden_dim = hidden,
        num_layers = 2,
        heads      = 4,
        K          = K,
        dropout    = 0.1,
    )

    z = model(batch)

    print(f"unified node embedding z : {z.shape}")   # (64, 64)
    print(f"value range              : [{z.min():.3f}, {z.max():.3f}]")
    print("Smoke test passed.")