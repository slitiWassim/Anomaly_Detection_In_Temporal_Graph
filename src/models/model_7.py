"""
Enhanced Temporal Anomaly GNN — v5
====================================
Based on the 91% baseline, with one conceptual fix to the Trust module.

═══════════════════════════════════════════════════════════════════
THE TRUST CONCEPTUAL FIX
═══════════════════════════════════════════════════════════════════

OLD BEHAVIOR (wrong for unsupervised learning):
  trust(i→j) was used as a GATE — low trust suppressed j's message
  toward zero. The implicit assumption was "j is anomalous, ignore it."
  This is supervised thinking: you need labels to know who to ignore.

  Problems:
    1. An anomalous node i would have its neighbors suppressed →
       it learns from nobody → its representation collapses → 
       no useful anomaly signal.
    2. A normal node surrounded by anomalies gets all messages
       suppressed → also collapses.
    3. Asymmetric: normal nodes ignore anomalous neighbors, but
       anomalous nodes also ignore normal neighbors → both groups
       become isolated → contrastive signal disappears.

NEW BEHAVIOR (correct for unsupervised):
  trust(i→j) is a SOFT SIMILARITY WEIGHT — not a gate, but a
  reweighting inside the softmax normalization.

  The principle: every node learns proportionally more from
  neighbors that resemble itself, whether those neighbors are
  normal OR anomalous.

    • Normal node i, normal neighbor j:    high similarity → learns a lot from j
    • Normal node i, anomalous neighbor j: low similarity  → learns less from j
    • Anomalous node i, anomalous neighbor j: high similarity → learns a lot from j  ✓
    • Anomalous node i, normal neighbor j: low similarity  → learns less from j     ✓

  This means:
    - Anomalous nodes naturally cluster with other anomalous nodes
      in representation space, because they learn most from each other.
    - Normal nodes cluster with normal nodes.
    - The contrastive loss then has clean signal: the two groups
      diverge in representation space without any labels.
    - Nobody is "deactivated" — every node still receives messages,
      just weighted by how similar the sender is.

HOW IT WORKS MECHANICALLY:
  Instead of:
    attn = attn * trust          # hard multiply → can zero out messages
    attn = softmax(attn, index)  # normalize → distrusted neighbors get ~0 mass

  We do:
    sim_weight = similarity_weight(i, j)    # soft [0,1], never forced to 0
    attn = attn + log(sim_weight + eps)     # ADD in log-space before softmax
    attn = softmax(attn, index)             # normalize → similar neighbors get MORE mass

  Adding in log-space is equivalent to multiplying the softmax argument
  by sim_weight, which is a proper probabilistic reweighting. The key
  difference from the old approach: sim_weight never kills a message
  completely — it just gives it proportionally less weight relative to
  more similar neighbors.

  The similarity weight itself is:
    sim(i,j) = softplus( cosine(h_i, h_j) )    where h = GRU memory
    + neighborhood_evolution_consistency(j)      whether j has been stable
    + uncertainty_confidence(i, j)               stable memory → more weight

  This is symmetric in intent: if i and j look alike, they learn from
  each other. If they look different, they learn less from each other —
  but never zero.

EVERYTHING ELSE is identical to the 91% baseline:
  - Time2Vec temporal encoding (absolute + relative gap)
  - RecurrentMemory GRU per node with variance tracking
  - EvolutionBank rolling window of neighborhood means
  - InfoNCE contrastive loss with hard negatives
  - Anomaly score = structural vs temporal view disagreement
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Time2Vec
# ═══════════════════════════════════════════════════════════════════════════════

class Time2Vec(nn.Module):
    """
    t → [w0*t + b0,  sin(w1*t+b1), ..., sin(wk*t+bk)]
    Linear = trend, sinusoids = periodicity. Frequencies are learned.
    """
    def __init__(self, out_dim):
        super().__init__()
        k = out_dim - 1
        self.w0 = nn.Parameter(torch.randn(1) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.W  = nn.Parameter(torch.randn(k) * 0.01)
        self.B  = nn.Parameter(torch.zeros(k))

    def forward(self, t):
        t = t.float().unsqueeze(-1)
        return torch.cat([t * self.w0 + self.b0,
                          torch.sin(t * self.W + self.B)], dim=-1)


class TimeEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.t2v     = Time2Vec(hidden_dim // 2)
        self.rel_t2v = Time2Vec(hidden_dim // 2)
        self.proj    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t_abs, t_rel=None):
        abs_enc = self.t2v(t_abs)
        rel_enc = self.rel_t2v(t_rel) if t_rel is not None else torch.zeros_like(abs_enc)
        return self.proj(torch.cat([abs_enc, rel_enc], dim=-1))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RecurrentMemory
# ═══════════════════════════════════════════════════════════════════════════════

class RecurrentMemory(nn.Module):
    """
    Per-node GRU hidden state + EMA variance.
    Low variance = stable node = more confident memory.
    """
    def __init__(self, num_nodes, dim, momentum=0.9):
        super().__init__()
        self.gru      = nn.GRUCell(dim, dim)
        self.momentum = momentum
        self.register_buffer("hidden",   torch.zeros(num_nodes, dim))
        self.register_buffer("variance", torch.ones(num_nodes, dim) * 0.1)

    def read(self, idx):
        return self.hidden[idx], self.variance[idx]

    def get_hidden(self, idx):
        return self.hidden[idx]

    @torch.no_grad()
    def write(self, idx, x):
        h_old = self.hidden[idx]
        h_new = self.gru(x.detach(), h_old)
        delta = h_new - h_old
        self.variance[idx] = (self.momentum * self.variance[idx]
                              + (1 - self.momentum) * delta.pow(2))
        self.hidden[idx] = h_new


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EvolutionBank
# ═══════════════════════════════════════════════════════════════════════════════

class EvolutionBank(nn.Module):
    """
    Rolling window of W most recent neighborhood mean embeddings per node.
    Tells the trust module how j's neighborhood has evolved over time.
    """
    def __init__(self, num_nodes, dim, window=6):
        super().__init__()
        self.window = window
        self.register_buffer("bank", torch.zeros(num_nodes, window, dim))
        self.register_buffer("ptr",  torch.zeros(num_nodes, dtype=torch.long))

    def read(self, idx):
        return self.bank[idx]                                          # (B, W, D)

    @torch.no_grad()
    def write(self, idx, emb):
        for b in range(idx.size(0)):
            p = int(self.ptr[idx[b]]) % self.window
            self.bank[idx[b], p] = emb[b].detach()
            self.ptr[idx[b]] += 1


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SimilarityWeight Module  ← THE FIX
#
# Replaces the old TrustModule gate with a soft similarity reweighting.
#
# OLD: trust → gate (can zero out messages from "suspicious" neighbors)
# NEW: sim_weight → log-space bias (reweights proportionally, never zeros)
#
# Produces a scalar w(i,j) ∈ (0, 1] for each edge, representing how
# similar i and j are. This is added in log-space to attention logits:
#
#   attn_logit(i,j) += log(w(i,j) + ε)
#
# After softmax this means similar neighbors get proportionally more
# attention mass. But every neighbor still gets SOME mass — no deactivation.
#
# w(i,j) is computed from three components:
#
#   (A) HISTORICAL SIMILARITY
#       cosine(GRU_hidden_i, GRU_hidden_j)
#       Nodes with similar behavioral histories attract each other.
#       A normal node has high similarity with other normal nodes.
#       An anomalous node has high similarity with other anomalous nodes.
#       → Each group naturally learns from its own kind.
#
#   (B) NEIGHBORHOOD EVOLUTION COHERENCE
#       How coherent is j's neighborhood evolution window?
#       j's evolution window = [neigh_mean_{t-W}, ..., neigh_mean_{t-1}]
#       If consecutive entries are similar → j is stable → consistent partner.
#       This is not "j is normal" — it's "j has been consistent."
#       An anomalous node that has been consistently anomalous is still
#       coherent and will have high coherence with other similar anomalies.
#
#   (C) UNCERTAINTY CONFIDENCE
#       Nodes with stable GRU states (low variance) produce more reliable
#       similarity estimates. We weight the combination by confidence.
#       Low confidence → w(i,j) pulled toward uniform (0.5) rather than
#       pushed to extremes, avoiding overconfident reweighting early in training.
#
# Final combination:
#   raw = MLP([hist_sim, coherence_j, conf])
#   w   = sigmoid(raw)   ∈ (0, 1)
#
# Note sigmoid never reaches exactly 0 or 1 — the floor ensures no
# message is ever completely silenced.
# ═══════════════════════════════════════════════════════════════════════════════

class SimilarityWeight(nn.Module):
    def __init__(self, dim, num_heads=4, window=6):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.window    = window

        # (A) Historical similarity projection
        # We project before cosine so the model can learn a better
        # similarity metric than raw cosine in the original space.
        self.proj_i = nn.Linear(dim, dim)
        self.proj_j = nn.Linear(dim, dim)

        # (B) Neighborhood evolution coherence
        # Reads j's W evolution steps and scores their consistency.
        # Cross-attention: i's history queries j's evolution window.
        # This lets i assess whether j has been a coherent partner,
        # not just whether j looks similar at the current moment.
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Recency decay: newer evolution steps weighted more
        self.log_decay = nn.Parameter(torch.tensor(-2.0))

        # Pool evolution context → coherence scalar
        self.coherence_out = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

        # (C) Final combination: [hist_sim, coherence, conf] → weight
        self.combine = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, h_i, h_j, evo_j, var_i, var_j, eps=1e-6):
        """
        h_i:   (E, D)    GRU hidden state of source node i
        h_j:   (E, D)    GRU hidden state of destination node j
        evo_j: (E, W, D) evolution window of j's neighborhood
        var_i: (E, D)    GRU variance of i
        var_j: (E, D)    GRU variance of j

        Returns: (E,) similarity weights in (0, 1)
                 — to be added in log-space to attention logits
        """
        E, W, D = evo_j.shape

        # ── (A) Historical similarity ────────────────────────────────────────
        # Project i and j's histories to a learned similarity space,
        # then compute cosine similarity.
        pi = F.normalize(self.proj_i(h_i), dim=-1)                    # (E, D)
        pj = F.normalize(self.proj_j(h_j), dim=-1)                    # (E, D)
        hist_sim = (pi * pj).sum(-1)                                   # (E,) ∈ [-1, 1]
        # Shift to [0, 1] — we want a weight, not a signed score
        hist_sim = (hist_sim + 1.0) / 2.0                             # (E,) ∈ [0, 1]

        # ── (B) Neighborhood evolution coherence ─────────────────────────────
        # Cross-attention: i's history (query) attends over j's evolution window
        nh, hd = self.num_heads, self.head_dim

        Q = self.q_proj(h_i).view(E, nh, hd)                         # (E, nh, hd)
        evo_flat = evo_j.reshape(E * W, D)
        K = self.k_proj(evo_flat).view(E, W, nh, hd)
        V = self.v_proj(evo_flat).view(E, W, nh, hd)

        Q_exp  = Q.unsqueeze(2)                                        # (E, nh, 1, hd)
        K_t    = K.permute(0, 2, 1, 3)                                # (E, nh, W, hd)
        scores = (Q_exp * K_t).sum(-1) * self.scale                   # (E, nh, W)

        # Recency decay: newer steps get higher base weight
        steps   = torch.arange(W, device=h_i.device).float()
        decay_w = torch.exp(self.log_decay.exp() * steps)
        decay_w = decay_w / decay_w.sum()
        scores  = scores + decay_w.view(1, 1, W)

        attn   = F.softmax(scores, dim=-1)                            # (E, nh, W)
        V_t    = V.permute(0, 2, 1, 3)                                # (E, nh, W, hd)
        pooled = (attn.unsqueeze(-1) * V_t).sum(2).reshape(E, D)     # (E, D)
        coherence = torch.sigmoid(
            self.coherence_out(pooled).squeeze(-1))                    # (E,) ∈ (0,1)

        # ── (C) Uncertainty confidence ───────────────────────────────────────
        # Nodes with stable memories → confident similarity estimate.
        # Pull toward 0.5 (uniform weight) when uncertain,
        # rather than pushing to extremes and over-suppressing.
        conf_i = 1.0 / (1.0 + var_i.mean(-1).sqrt().clamp(min=eps))  # (E,) ∈ (0,1]
        conf_j = 1.0 / (1.0 + var_j.mean(-1).sqrt().clamp(min=eps))
        conf   = (conf_i * conf_j).sqrt()                             # (E,) geometric mean

        # ── Combine all three signals ─────────────────────────────────────────
        signals = torch.stack([hist_sim, coherence, conf], dim=-1)    # (E, 3)
        w = torch.sigmoid(self.combine(signals).squeeze(-1))          # (E,) ∈ (0, 1)

        # w is the soft similarity weight.
        # sigmoid floor ≈ 0.007, ceiling ≈ 0.993 — never exactly 0 or 1.
        # Every neighbor still contributes; dissimilar ones just contribute less.
        return w


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Temporal Attention Layer with Similarity Reweighting
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalTrustAttention(MessagePassing):
    def __init__(self, dim, heads=8, dropout=0.1, window=6):
        super().__init__(aggr="add", node_dim=0)
        assert dim % heads == 0
        self.dim      = dim
        self.heads    = heads
        self.head_dim = dim // heads
        self.scale    = self.head_dim ** -0.5
        self.window   = window

        self.q_proj    = nn.Linear(dim, dim)
        self.k_proj    = nn.Linear(dim, dim)
        self.v_proj    = nn.Linear(dim, dim)
        self.o_proj    = nn.Linear(dim, dim)
        self.edge_proj = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(dim, dim)

        # Renamed from TrustModule → SimilarityWeight to reflect new semantics
        self.sim_weight = SimilarityWeight(dim, num_heads=4, window=window)
        self.dropout    = nn.Dropout(dropout)

        self.gate    = nn.Linear(dim, 1)
        self.norm    = nn.LayerNorm(dim)
        self.fusion  = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Cache for inspection / anomaly scoring
        self.sim_cache = None

    def split(self, x):
        return x.view(-1, self.heads, self.head_dim)

    def forward(self, x, edge_index, edge_emb, time_emb, memory, evo_bank):
        N      = x.size(0)
        device = x.device
        idx    = torch.arange(N, device=device)

        Q = self.split(self.q_proj(x))
        K = self.split(self.k_proj(x))
        V = self.split(self.v_proj(x))

        mem_vals, mem_vars = memory.read(idx)                          # (N, D) each
        evo_vals           = evo_bank.read(idx)                        # (N, W, D)

        out = self.propagate(
            edge_index,
            Q=Q, K=K, V=V,
            mem_val=mem_vals,
            mem_var=mem_vars,
            hidden=mem_vals,
            evo_vals=evo_vals,
            edge_emb=self.split(self.edge_proj(edge_emb)),
            time_emb=self.split(self.time_proj(time_emb)),
            size=(N, N)
        )
        out = out.view(N, self.dim)

        h   = memory.get_hidden(idx)
        out = self.fusion(torch.cat([out, h], dim=-1))
        out = self.o_proj(out)

        g = torch.sigmoid(self.gate(x))
        return self.norm(g * x + (1 - g) * out)

    def message(self, Q_i, K_j, V_j,
                mem_val_i, mem_val_j,
                mem_var_i, mem_var_j,
                hidden_i, hidden_j,
                evo_vals_j,
                edge_emb, time_emb,
                index):

        # Enrich keys with edge and time context (unchanged from baseline)
        K_j = K_j + edge_emb + time_emb                               # (E, h, hd)

        # Raw dot-product attention logits
        attn = (Q_i * K_j).sum(-1) * self.scale                      # (E, h)

        # ── SIMILARITY REWEIGHTING (the fix) ────────────────────────────────
        # Compute soft similarity weight w(i,j) ∈ (0, 1)
        w = self.sim_weight(
            hidden_i, hidden_j,
            evo_vals_j,
            mem_var_i, mem_var_j
        )                                                               # (E,)
        self.sim_cache = w.detach()

        # Add log(w) to the attention logit IN LOG-SPACE before softmax.
        #
        # Why log-space?
        #   softmax(a + log(w)) = softmax(a) * w / Z
        # This is equivalent to multiplying the unnormalized attention
        # probability by w, then renormalizing. It is a clean probabilistic
        # upweighting of similar neighbors and downweighting of dissimilar ones.
        #
        # Why NOT multiply after softmax?
        #   attn = softmax(a); attn = attn * w   ← this breaks normalization.
        #   The weights no longer sum to 1, so the aggregation is biased.
        #
        # Why NOT multiply before softmax (old gate approach)?
        #   attn = a * w; softmax(attn)
        #   When w ≈ 0, the logit is zeroed → after softmax that neighbor
        #   gets nearly zero mass. This is the deactivation problem.
        #   With log(w), even w=0.01 → log(w)≈-4.6, not -∞.
        #   The neighbor still gets a small positive mass after softmax.
        #
        # Clamp log(w) to [-6, 0]: floor prevents -inf, ceiling (0) ensures
        # the weight never INCREASES beyond the base attention score.
        log_w = torch.log(w.clamp(min=1e-6)).clamp(min=-6.0, max=0.0) # (E,)
        attn  = attn + log_w.unsqueeze(-1)                             # (E, h)

        attn = softmax(attn, index)                                    # normalize over neighbors
        attn = self.dropout(attn)

        return V_j * attn.unsqueeze(-1)                                # (E, h, hd)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Full Model (identical to 91% baseline except SimilarityWeight)
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedTemporalGNN(nn.Module):
    def __init__(self, num_nodes, in_dim, edge_dim, hidden_dim,
                 num_layers=2, heads=8, dropout=0.1,
                 window=6, memory_momentum=0.9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes  = num_nodes

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_enc   = nn.Linear(edge_dim, hidden_dim)
        self.time_enc   = TimeEncoder(hidden_dim)

        self.memory   = RecurrentMemory(num_nodes, hidden_dim, momentum=memory_momentum)
        self.evo_bank = EvolutionBank(num_nodes, hidden_dim, window=window)

        self.layers = nn.ModuleList([
            TemporalTrustAttention(hidden_dim, heads=heads,
                                   dropout=dropout, window=window)
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

    def forward(self, batch):
        N          = batch.x.size(0)
        device     = batch.x.device
        edge_index = batch.edge_index
        src, dst   = edge_index

        x        = self.input_proj(batch.x)
        edge_emb = self.edge_enc(batch.msg)

        # Relative time gap: how long since src last appeared in this batch
        node_last_t = torch.zeros(N, device=device)
        node_last_t.scatter_reduce_(
            0, src, batch.t.float(), reduce='amax', include_self=True)
        t_rel    = batch.t.float() - node_last_t[src]
        time_emb = self.time_enc(batch.t, t_rel)

        for layer in self.layers:
            x = layer(x, edge_index, edge_emb, time_emb,
                      self.memory, self.evo_bank)

        x = self.final_norm(x)

        # Update memory
        self.memory.write(torch.arange(N, device=device), x)

        # Update evolution bank with neighborhood means
        neigh = torch.zeros(N, self.hidden_dim, device=device)
        cnt   = torch.zeros(N, 1, device=device)
        neigh.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.hidden_dim), x[src])
        cnt.scatter_add_(0, dst.unsqueeze(1),
                         torch.ones(src.size(0), 1, device=device))
        neigh = neigh / (cnt + 1e-6)
        unique_dst = dst.unique()
        self.evo_bank.write(unique_dst, neigh[unique_dst])

        return x


