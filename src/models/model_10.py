"""
Temporal Trust GNN for Unsupervised Anomaly Detection
=====================================================
Produces node representations for downstream contrastive learning.
No labels required — fully unsupervised.

Trust Mechanism  trust(i->j) in (0, 2):
  For every edge (i, j) in the current snapshot:

  1. K-VIEW SNAPSHOTS
     The EvolutionBank stores the last K activity snapshots for every node.
     Each snapshot contains: (node_embedding, edge_features_mean, timestamp).
     These were computed at the time of each past activity using a shared
     lightweight GNN (SnapshotGNN) that sees node features + edge features
     + time encoding at that moment.

  2. TIME POSITIONAL ENCODING
     Each of the K snapshot embeddings is enriched with a Time2Vec encoding
     of its timestamp. This makes each view time-stamped so that the
     comparison is aware of WHEN each activity happened, not just WHAT.

  3. K-VIEW TRAJECTORY SIMILARITY
     The K time-enriched views of node i are compared with the K
     time-enriched views of node j using mean cosine similarity across
     the K positions:
         traj_sim = (1/K) * sum_k  cos(view_i_k, view_j_k)
     This captures: "have i and j been evolving in a similar way
     over their last K activities?"

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
from torch_geometric.nn import MessagePassing
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
# 4.  SnapshotGNN  — lightweight GNN that produces a node embedding
#                   from ALL available features at a given activity moment
#
#  At each of a node's K past activities, this GNN was used to compute
#  the node's embedding by combining:
#    - the node's own feature vector at that time
#    - the mean of edge features of interactions at that time
#    - the time encoding of that activity's timestamp
#  All projected and fused into a single snapshot embedding.
#
#  We store the OUTPUT of this GNN in the EvolutionBank, not the raw inputs.
#  So the bank holds K rich, contextual embeddings per node.
# ═══════════════════════════════════════════════════════════════════════════════

class SnapshotGNN(nn.Module):
    """
    Computes a single snapshot embedding for a node at one activity step.

    Inputs (all at the time of this activity):
        x_node  : (B, dim)   node's current projected feature
        e_mean  : (B, dim)   mean of edge features for this node's edges
        t_enc   : (B, dim)   Time2Vec encoding of this activity's timestamp

    Output: (B, dim)  rich contextual snapshot embedding
    """
    def __init__(self, dim: int):
        super().__init__()
        # Fuse node features + edge context + time encoding
        self.fuse = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        # One-hop neighbourhood aggregation: refine with neighbour context
        self.aggr_proj  = nn.Linear(dim, dim)
        self.update_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self,
                x_node: torch.Tensor,   # (B, dim)
                e_mean: torch.Tensor,   # (B, dim)  mean edge features
                t_enc:  torch.Tensor,   # (B, dim)  time encoding
                ) -> torch.Tensor:      # (B, dim)
        # Fuse all three information sources
        fused = self.fuse(torch.cat([x_node, e_mean, t_enc], dim=-1))  # (B, dim)
        # Aggregate neighbour context (represented by e_mean as proxy)
        neigh = self.aggr_proj(e_mean)
        # Update: combine own fused state with neighbourhood context
        out = self.update_proj(torch.cat([fused, neigh], dim=-1))
        return self.norm(out)


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
# 6.  StructuralTrustModule  — CORE CONTRIBUTION
#
#  For every edge (i -> j), computes trust(i, j) in (0, 2).
#
#  Step-by-step:
#
#  [A] EMBEDDING SIMILARITY
#      cos(h_i, h_j) using current GRU hidden states.
#      Captures "are i and j playing similar roles right now?"
#
#  [B] DEGREE PROFILE SIMILARITY
#      1 - |deg_i - deg_j| / (deg_i + deg_j)
#      Captures "do i and j have similar connectivity patterns?"
#      Anomalous nodes (hubs, isolates, injectors) deviate here.
#
#  [C] K-VIEW TRAJECTORY SIMILARITY
#      Both i and j have K snapshot embeddings stored in EvolutionBank.
#      Each snapshot was produced by SnapshotGNN at that past activity.
#
#      Time Positional Encoding:
#          Each snapshot embedding e_k is enriched with Time2Vec(t_k):
#              view_k = Linear([e_k || Time2Vec(t_k)])
#          This makes the view aware of WHEN it was recorded.
#          Recent activities (larger t) have different encodings
#          from old activities, so the comparison is time-sensitive.
#
#      Trajectory Similarity:
#          After enriching, compare each position k:
#              traj_sim = (1/K) * sum_k  cos(view_i_k,  view_j_k)
#          This is a position-aligned comparison: the k-th most
#          recent activity of i is compared with the k-th most
#          recent activity of j.
#          If their trajectories have been evolving similarly
#          (same pattern of activity, at similar times),
#          traj_sim will be high.
#          Anomalous nodes will have deviant trajectory patterns
#          even if a single current embedding looks similar.
#
#  FUSION:
#      trust_logit = MLP([emb_cos, deg_sim, traj_sim])
#      trust       = 1 + tanh(trust_logit)   in (0, 2)
# ═══════════════════════════════════════════════════════════════════════════════

class StructuralTrustModule(nn.Module):
    def __init__(self, dim: int, window: int = 8, t2v_dim: int = 16):
        super().__init__()
        self.window  = window
        self.t2v_dim = t2v_dim

        # Time positional encoder for snapshot views
        self.t2v = Time2Vec(t2v_dim)

        # Project [snapshot_emb || time_enc] -> dim for each view
        self.view_proj = nn.Linear(dim + t2v_dim, dim)

        # Final MLP: 3 scalar signals -> trust logit
        h = max(dim // 4, 16)
        self.trust_mlp = nn.Sequential(
            nn.Linear(3, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

    def _encode_views(self,
                      embs:  torch.Tensor,   # (B, K, dim)
                      times: torch.Tensor,   # (B, K)
                      ) -> torch.Tensor:     # (B, K, dim)
        """
        Enrich each of the K snapshot embeddings with its time encoding.
        view_k = GELU( Linear([emb_k || Time2Vec(t_k)]) )
        """
        B, K, D = embs.shape
        # Encode all B*K timestamps in one call
        t_flat = times.reshape(B * K)                                # (B*K,)
        t_enc  = self.t2v(t_flat).view(B, K, self.t2v_dim)          # (B, K, t2v_dim)
        # Concatenate and project
        cat    = torch.cat([embs, t_enc], dim=-1)                    # (B, K, dim+t2v)
        return F.gelu(self.view_proj(cat))                           # (B, K, dim)

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

        # ── [A] Embedding similarity ─────────────────────────────────────────
        hi_n    = F.normalize(h_i, dim=-1, eps=eps)                  # (E, dim)
        hj_n    = F.normalize(h_j, dim=-1, eps=eps)
        emb_cos = (hi_n * hj_n).sum(-1, keepdim=True)               # (E, 1)

        # ── [B] Degree profile similarity ────────────────────────────────────
        di, dj  = deg_i.float(), deg_j.float()
        deg_sim = (1.0 - (di - dj).abs() / (di + dj + eps)
                   ).unsqueeze(-1)                                   # (E, 1)

        # ── [C] K-view trajectory similarity ─────────────────────────────────
        # Encode views: inject time positional encoding into each snapshot
        # We batch i and j together (stack on dim 0) to use one projection call
        views_i = self._encode_views(embs_i, t_i)                   # (E, K, dim)
        views_j = self._encode_views(embs_j, t_j)                   # (E, K, dim)

        # Normalize each view for cosine similarity
        vi_n = F.normalize(views_i, dim=-1, eps=eps)                 # (E, K, dim)
        vj_n = F.normalize(views_j, dim=-1, eps=eps)

        # Position-aligned cosine similarity at each of the K steps,
        # then average across K:
        #   traj_sim = mean_k cos(view_i_k, view_j_k)
        traj_sim = (vi_n * vj_n).sum(-1).mean(-1, keepdim=True)     # (E, 1) in [-1,1]

        # ── Fuse three signals ───────────────────────────────────────────────
        features    = torch.cat([emb_cos, deg_sim, traj_sim], dim=-1)  # (E, 3)
        trust_logit = self.trust_mlp(features).squeeze(-1)             # (E,)

        # ── trust in (0, 2) ──────────────────────────────────────────────────
        #   trust > 1: similar neighbor     -> amplify learning
        #   trust < 1: dissimilar neighbor  -> suppress learning
        #   trust -> 0: anomalous neighbor  -> near-zero influence
        return 1.0 + torch.tanh(trust_logit)                         # (E,)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  TemporalTrustAttention  (one message-passing layer)
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
# 8.  EnhancedTemporalGNN  — full model
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

        # Snapshot GNN: produces the embedding stored in EvolutionBank
        # Called once per forward pass to compute the current snapshot
        self.snapshot_gnn = SnapshotGNN(hidden_dim)

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
        # For each node, aggregate its edge features and time encoding,
        # then pass through SnapshotGNN to get the snapshot embedding.
        # This embedding captures the node's state at THIS activity moment
        # using all available signals: node features + edge context + time.
        e_sum = torch.zeros(N, self.hidden_dim, device=device)
        e_cnt = torch.zeros(N, 1, device=device)
        e_sum.scatter_add_(0, src.unsqueeze(1).expand(-1, self.hidden_dim), edge_emb)
        e_cnt.scatter_add_(0, src.unsqueeze(1), torch.ones(src.size(0), 1, device=device))
        e_mean = e_sum / (e_cnt + 1e-6)                             # (N, hidden)

        # Mean time encoding per node
        t_sum = torch.zeros(N, self.hidden_dim, device=device)
        t_sum.scatter_add_(0, src.unsqueeze(1).expand(-1, self.hidden_dim), time_emb)
        t_mean = t_sum / (e_cnt + 1e-6)                             # (N, hidden)

        snapshot_emb = self.snapshot_gnn(x, e_mean, t_mean)        # (N, hidden)

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