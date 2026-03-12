import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Time2Vec + TimeEncoder
# ═══════════════════════════════════════════════════════════════════════════════

class Time2Vec(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        k = out_dim - 1
        self.w0 = nn.Parameter(torch.randn(1) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.W  = nn.Parameter(torch.randn(k) * 0.01)
        self.B  = nn.Parameter(torch.zeros(k))

    def forward(self, t):                                   # t: (N,)
        t        = t.float().unsqueeze(-1)                  # (N,1)
        linear   = t * self.w0 + self.b0                   # (N,1)
        periodic = torch.sin(t * self.W + self.B)          # (N,k)
        return torch.cat([linear, periodic], dim=-1)        # (N, out_dim)


class TimeEncoder(nn.Module):
    """
    Encodes absolute timestamp + inter-event delta (time since previous
    event for the same node).  Both are needed:
      - absolute t  → where in calendar time this event sits
      - delta t     → rhythm / burstiness of this node's activity
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.t2v_abs   = Time2Vec(hidden_dim // 2)
        self.t2v_delta = Time2Vec(hidden_dim // 2)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t_abs, t_delta=None):                 # both (E,)
        abs_enc   = self.t2v_abs(t_abs)
        delta_enc = self.t2v_delta(t_delta) if t_delta is not None \
                    else torch.zeros_like(abs_enc)
        return self.proj(torch.cat([abs_enc, delta_enc], dim=-1))   # (E, D)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Per-Node Activity Snapshot Builder
#
# KEY DESIGN DECISION:
#   Snapshots are built per node from that node's OWN interaction history.
#   Node i's K views reflect how i itself evolved, not global time bins.
#
#   Two strategies (controlled by `mode`):
#     'equal_count' — each view covers the same number of events
#                     good when activity is bursty / uneven
#     'equal_time'  — each view covers the same time span
#                     good when you care about calendar rhythm
#
#   Returns for each node:
#     view_edge_masks: list of K boolean masks over the node's local edges
#     view_t_centers:  (K,) representative timestamp per view
# ═══════════════════════════════════════════════════════════════════════════════

def build_per_node_snapshots(edge_index, edge_t, num_nodes, num_snapshots,
                              mode='equal_count'):
    """
    For every node i, partition its edges into K views.

    Returns:
        node_view_masks : list of length num_nodes, each entry is a
                          LongTensor of shape (K, E_i) — boolean mask
                          over that node's edges for each view.
                          We store as list[node] → list[view] → edge indices.
        node_view_centers: (num_nodes, K) — center timestamp of each view.
    """
    src = edge_index[0]
    E   = edge_index.size(1)

    # For each node, collect sorted edge indices
    # node_edges[i] = sorted indices into the global edge list
    node_edges = [[] for _ in range(num_nodes)]
    for e in range(E):
        node_edges[src[e].item()].append(e)
    # Also include edges where node is destination (it receives messages)
    dst = edge_index[1]
    for e in range(E):
        d = dst[e].item()
        if e not in node_edges[d]:        # avoid double-counting
            node_edges[d].append(e)

    node_view_edge_lists  = []   # node → [view_0_edges, view_1_edges, ...]
    node_view_centers     = torch.zeros(num_nodes, num_snapshots,
                                        device=edge_t.device)

    for i in range(num_nodes):
        edges_i = sorted(node_edges[i], key=lambda e: edge_t[e].item())
        views   = _split_edges_into_views(
            edges_i, edge_t, num_snapshots, mode
        )
        node_view_edge_lists.append(views)      # list of K lists of edge indices

        for k, v in enumerate(views):
            if len(v) > 0:
                node_view_centers[i, k] = edge_t[torch.tensor(v)].float().mean()
            else:
                # Empty view: inherit center from previous or default to 0
                node_view_centers[i, k] = (node_view_centers[i, k-1]
                                           if k > 0 else edge_t.float().mean())

    return node_view_edge_lists, node_view_centers


def _split_edges_into_views(sorted_edge_indices, edge_t, K, mode):
    """Split a sorted list of edge indices into K groups."""
    if len(sorted_edge_indices) == 0:
        return [[] for _ in range(K)]

    if mode == 'equal_count':
        # Each view gets ~equal number of events
        chunks = [[] for _ in range(K)]
        for idx, e in enumerate(sorted_edge_indices):
            k = min(int(idx * K / len(sorted_edge_indices)), K - 1)
            chunks[k].append(e)
        return chunks

    elif mode == 'equal_time':
        # Each view covers an equal time span
        t_vals  = [edge_t[e].item() for e in sorted_edge_indices]
        t_min, t_max = t_vals[0], t_vals[-1]
        if t_min == t_max:
            t_max = t_min + 1.0
        bins   = [t_min + (t_max - t_min) * k / K for k in range(K + 1)]
        chunks = [[] for _ in range(K)]
        for e, tv in zip(sorted_edge_indices, t_vals):
            k = min(int((tv - t_min) / (t_max - t_min + 1e-9) * K), K - 1)
            chunks[k].append(e)
        return chunks

    else:
        raise ValueError(f"Unknown mode {mode}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Recurrent Node Memory with Uncertainty
# ═══════════════════════════════════════════════════════════════════════════════

class RecurrentMemory(nn.Module):
    def __init__(self, num_nodes, dim, momentum=0.9):
        super().__init__()
        self.gru      = nn.GRUCell(dim, dim)
        self.momentum = momentum
        self.register_buffer("hidden",   torch.zeros(num_nodes, dim))
        self.register_buffer("variance", torch.ones(num_nodes, dim) * 0.1)
        self.register_buffer("last_t",   torch.zeros(num_nodes))

    def read(self, idx):
        return self.hidden[idx], self.variance[idx]

    def get_hidden(self, idx):
        return self.hidden[idx]

    @torch.no_grad()
    def write(self, idx, x, t=None):
        h_old = self.hidden[idx]
        h_new = self.gru(x.detach(), h_old)
        delta = h_new - h_old
        self.variance[idx] = (self.momentum * self.variance[idx]
                              + (1 - self.momentum) * delta.pow(2))
        self.hidden[idx] = h_new
        if t is not None:
            self.last_t[idx] = t.float()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Neighborhood Evolution Bank
# ═══════════════════════════════════════════════════════════════════════════════

class EvolutionBank(nn.Module):
    def __init__(self, num_nodes, dim, window=6):
        super().__init__()
        self.window = window
        self.register_buffer("bank", torch.zeros(num_nodes, window, dim))
        self.register_buffer("ptr",  torch.zeros(num_nodes, dtype=torch.long))

    def read(self, idx):
        return self.bank[idx]

    @torch.no_grad()
    def write(self, idx, emb):
        for b in range(idx.size(0)):
            p = int(self.ptr[idx[b]]) % self.window
            self.bank[idx[b], p] = emb[b].detach()
            self.ptr[idx[b]] += 1


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Trust Module
# ═══════════════════════════════════════════════════════════════════════════════

class TrustModule(nn.Module):
    def __init__(self, dim, num_heads=4, window=6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.window    = window
        self.q_proj    = nn.Linear(dim, dim)
        self.k_proj    = nn.Linear(dim, dim)
        self.v_proj    = nn.Linear(dim, dim)
        self.log_decay = nn.Parameter(torch.tensor(-2.0))
        self.trust_out = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        )

    def forward(self, h_i, evo_j, var_i, var_j, eps=1e-6):
        E, W, D = evo_j.shape
        nh, hd  = self.num_heads, self.head_dim
        Q = self.q_proj(h_i).view(E, nh, hd)
        K = self.k_proj(evo_j.reshape(E * W, D)).view(E, W, nh, hd)
        V = self.v_proj(evo_j.reshape(E * W, D)).view(E, W, nh, hd)
        scores = (Q.unsqueeze(2) * K.permute(0,2,1,3)).sum(-1) * self.scale
        steps   = torch.arange(W, device=h_i.device).float()
        decay_w = torch.exp(self.log_decay.exp() * steps)
        scores  = scores + (decay_w / decay_w.sum()).view(1, 1, W)
        attn    = F.softmax(scores, dim=-1)
        pooled  = (attn.unsqueeze(-1) * V.permute(0,2,1,3)).sum(2).reshape(E, D)
        trust_logit = self.trust_out(pooled).squeeze(-1)
        conf_i = 1.0 / (1.0 + var_i.mean(-1).sqrt().clamp(min=eps))
        conf_j = 1.0 / (1.0 + var_j.mean(-1).sqrt().clamp(min=eps))
        conf   = (conf_i * conf_j).sqrt()
        return torch.sigmoid(trust_logit + conf.log().clamp(min=-5))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Similarity-Gated Trust Attention Layer
#    Applied independently to each snapshot view's edge set.
# ═══════════════════════════════════════════════════════════════════════════════

class SimilarityGatedTrustAttention(MessagePassing):
    def __init__(self, dim, heads=8, dropout=0.1, window=6, sim_temp=0.5):
        super().__init__(aggr="add", node_dim=0)
        self.dim, self.heads = dim, heads
        self.head_dim = dim // heads
        self.scale    = self.head_dim ** -0.5
        self.window   = window
        self.sim_temp = sim_temp

        self.q_proj    = nn.Linear(dim, dim)
        self.k_proj    = nn.Linear(dim, dim)
        self.v_proj    = nn.Linear(dim, dim)
        self.o_proj    = nn.Linear(dim, dim)
        self.edge_proj = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(dim, dim)

        self.trust       = TrustModule(dim, num_heads=4, window=window)
        self.dropout     = nn.Dropout(dropout)
        self.gate_balance = nn.Parameter(torch.tensor(0.5))
        self.gate        = nn.Linear(dim, 1)
        self.norm        = nn.LayerNorm(dim)
        self.fusion      = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def split(self, x):
        return x.view(-1, self.heads, self.head_dim)

    def forward(self, x, edge_index, edge_emb, time_emb, memory, evo_bank):
        """
        x:           (N, D)  current node features
        edge_index:  (2, E)  edges in THIS snapshot (already filtered)
        edge_emb:    (E, D)
        time_emb:    (E, D)
        Returns updated (N, D)
        """
        N      = x.size(0)
        device = x.device

        Q = self.split(self.q_proj(x))
        K = self.split(self.k_proj(x))
        V = self.split(self.v_proj(x))

        mem_vals, mem_vars = memory.read(torch.arange(N, device=device))
        evo_vals           = evo_bank.read(torch.arange(N, device=device))

        out = self.propagate(
            edge_index,
            Q=Q, K=K, V=V, x=x,
            mem_val=mem_vals, mem_var=mem_vars,
            hidden=mem_vals, evo_vals=evo_vals,
            edge_emb=self.split(self.edge_proj(edge_emb)),
            time_emb=self.split(self.time_proj(time_emb)),
            size=(N, N)
        )

        out = out.view(N, self.dim)
        h   = memory.get_hidden(torch.arange(N, device=device))
        out = self.fusion(torch.cat([out, h], dim=-1))
        out = self.o_proj(out)
        g   = torch.sigmoid(self.gate(x))
        return self.norm(g * x + (1 - g) * out)

    def message(self, Q_i, K_j, V_j, x_i, x_j,
                mem_var_i, mem_var_j, hidden_i, evo_vals_j,
                edge_emb, time_emb, index):
        K_j  = K_j + edge_emb + time_emb
        attn = (Q_i * K_j).sum(-1) * self.scale                   # (E, heads)

        trust = self.trust(hidden_i, evo_vals_j, mem_var_i, mem_var_j)
        sim   = torch.sigmoid(
            F.cosine_similarity(x_i, x_j, dim=-1) / self.sim_temp
        )
        alpha    = torch.sigmoid(self.gate_balance)
        combined = alpha * trust + (1 - alpha) * sim               # (E,)

        attn = softmax(attn * combined.unsqueeze(-1), index)
        attn = self.dropout(attn)
        return V_j * attn.unsqueeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Per-Node View Encoder
#
# Given a node i and its K activity windows, this applies the attention layers
# to each window IN SEQUENCE, carrying the hidden state forward between views.
#
# Crucially, each view's time encoding uses:
#   t_abs   = absolute timestamp of each edge
#   t_delta = time since previous event FOR THIS NODE
#             (not a global delta — it's node i's personal inter-event gap)
#
# This captures each node's OWN temporal rhythm.
# ═══════════════════════════════════════════════════════════════════════════════

class PerNodeViewEncoder(nn.Module):
    """
    Runs attention layers on each of node i's K activity views.
    Carries x forward between views so later views build on earlier context.
    Returns (N, K, D) — one embedding per view per node.
    """
    def __init__(self, layers, edge_enc, time_enc):
        super().__init__()
        # These are shared with the parent model (not re-instantiated)
        self.layers    = layers
        self.edge_enc  = edge_enc
        self.time_enc  = time_enc

    def forward(self, x_init, node_view_edge_lists, node_view_centers,
                full_edge_index, full_t, full_msg, memory, evo_bank, num_snapshots):
        """
        x_init:                (N, D)
        node_view_edge_lists:  list[N] → list[K] → list of global edge indices
        node_view_centers:     (N, K) — center timestamp of each node's views
        full_edge_index:       (2, E_total)
        full_t:                (E_total,)
        full_msg:              (E_total, edge_dim) — raw edge features before encoding

        Returns:
            view_embs: (N, K, D)
        """
        N      = x_init.size(0)
        K      = num_snapshots
        device = x_init.device
        D      = x_init.size(-1)

        view_embs = torch.zeros(N, K, D, device=device)

        # We process all K views together in a batch-of-snapshots loop.
        # For each view k, we build a sub-graph containing only the edges
        # that belong to view k for at least one node.
        # This is more efficient than looping over individual nodes.

        x_current = x_init.clone()   # carries state across views

        for k in range(K):
            # Collect all edges that are in view k for any node
            view_k_edge_set = set()
            for i in range(N):
                view_k_edge_set.update(node_view_edge_lists[i][k])
            view_k_edges = sorted(view_k_edge_set)

            if len(view_k_edges) == 0:
                view_embs[:, k, :] = x_current
                continue

            eidx  = torch.tensor(view_k_edges, device=device, dtype=torch.long)
            sub_ei = full_edge_index[:, eidx]        # (2, E_k)
            sub_t  = full_t[eidx]                    # (E_k,)
            sub_msg = full_msg[eidx]                 # (E_k, edge_dim)

            # Per-edge inter-event delta:
            # For each edge, compute time since the previous edge of the same source
            # This is the node's personal activity rhythm, not a global delta
            src_nodes  = sub_ei[0]
            t_delta    = _compute_inter_event_delta(src_nodes, sub_t, N, device)

            edge_emb = self.edge_enc(sub_msg)                     # (E_k, D)
            time_emb = self.time_enc(sub_t, t_delta)              # (E_k, D)

            # Run all attention layers on this snapshot
            x_view = x_current.clone()
            for layer in self.layers:
                x_view = layer(x_view, sub_ei, edge_emb, time_emb, memory, evo_bank)

            view_embs[:, k, :] = x_view
            # Carry forward: next view starts from the updated state
            x_current = x_view

        return view_embs


def _compute_inter_event_delta(src_nodes, edge_t, num_nodes, device):
    """
    For each edge e with source node src[e] and timestamp t[e],
    compute t[e] - t[prev_edge_of_src[e]].
    First event per node gets delta=0.
    This measures the personal inter-event gap of each source node.
    """
    E      = src_nodes.size(0)
    delta  = torch.zeros(E, device=device)
    last_t = torch.full((num_nodes,), -1.0, device=device)

    for e in range(E):
        s = src_nodes[e].item()
        t = edge_t[e].float().item()
        if last_t[s] >= 0:
            delta[e] = t - last_t[s]
        last_t[s] = t

    return delta


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Temporal View Integrator (GRU + attention over K views)
#
# After getting K per-node views, this module integrates them into
# a single final representation.
#
# The GRU explicitly receives the inter-VIEW time delta:
#   Δt_k = center_time(view_k) - center_time(view_{k-1})
# This is PER NODE: node i's Δt_k reflects how much of node i's
# personal time elapsed between its k-1'th and k'th activity windows.
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalViewIntegrator(nn.Module):
    def __init__(self, dim, num_views):
        super().__init__()
        self.dim       = dim
        self.num_views = num_views
        self.t2v_delta  = Time2Vec(dim // 2)
        self.delta_proj = nn.Linear(dim // 2, dim)
        self.gru        = nn.GRUCell(dim * 2, dim)
        self.view_attn  = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.out_norm   = nn.LayerNorm(dim)

    def forward(self, view_embs, node_view_centers):
        """
        view_embs:         (N, K, D)
        node_view_centers: (N, K)  — each node's own view timestamps
        Returns:           (N, D)
        """
        N, K, D = view_embs.shape
        device  = view_embs.device

        # Inter-view time deltas — PER NODE
        # shape (N, K): Δt[i, k] = center[i,k] - center[i,k-1]
        view_deltas        = torch.zeros(N, K, device=device)
        view_deltas[:, 1:] = (node_view_centers[:, 1:]
                              - node_view_centers[:, :-1]).clamp(min=0)

        h = torch.zeros(N, D, device=device)
        gru_out = []

        for k in range(K):
            # Encode each node's individual inter-view gap at step k
            delta_enc = self.delta_proj(
                self.t2v_delta(view_deltas[:, k])           # (N, D/2) → (N, D)
            )
            inp = torch.cat([view_embs[:, k, :], delta_enc], dim=-1)   # (N, 2D)
            h   = self.gru(inp, h)
            gru_out.append(h)

        gru_seq = torch.stack(gru_out, dim=1)                # (N, K, D)

        # CLS token attends over all K views to weight temporal moments
        cls    = self.cls_token.expand(N, -1, -1)            # (N, 1, D)
        seq    = torch.cat([cls, gru_seq], dim=1)            # (N, K+1, D)
        out, _ = self.view_attn(seq, seq, seq)
        return self.out_norm(out[:, 0, :])                   # (N, D)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Full Model
# ═══════════════════════════════════════════════════════════════════════════════

class MultiViewTemporalGNN(nn.Module):
    """
    Full pipeline:
      1. Build per-node activity snapshots (each node's OWN K windows)
      2. Encode each snapshot view with SimilarityGatedTrustAttention
         — trust + similarity gates applied at EVERY view
         — state carried forward between views
      3. Integrate K views with GRU using per-node inter-view time deltas
      4. CLS attention pooling → final (N, D) representation
      5. Update recurrent memory + evolution bank
    """
    def __init__(self, num_nodes, in_dim, edge_dim, hidden_dim,
                 num_layers=2, heads=8, dropout=0.1,
                 num_snapshots=6, window=6, memory_momentum=0.9,
                 sim_temp=0.5, snapshot_mode='equal_count'):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.num_nodes     = num_nodes
        self.num_snapshots = num_snapshots
        self.snapshot_mode = snapshot_mode

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_enc   = nn.Linear(edge_dim, hidden_dim)
        self.time_enc   = TimeEncoder(hidden_dim)

        self.memory   = RecurrentMemory(num_nodes, hidden_dim, momentum=memory_momentum)
        self.evo_bank = EvolutionBank(num_nodes, hidden_dim, window=window)

        self.layers = nn.ModuleList([
            SimilarityGatedTrustAttention(
                hidden_dim, heads=heads, dropout=dropout,
                window=window, sim_temp=sim_temp
            )
            for _ in range(num_layers)
        ])

        self.view_encoder = PerNodeViewEncoder(
            self.layers, self.edge_enc, self.time_enc
        )

        self.view_integrator = TemporalViewIntegrator(hidden_dim, num_snapshots)

        self.final_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch):
        """
        batch.x          : (N, in_dim)
        batch.edge_index : (2, E)
        batch.t          : (E,)   edge timestamps
        batch.msg        : (E, edge_dim) raw edge features
        """
        N      = batch.x.size(0)
        device = batch.x.device

        x = self.input_proj(batch.x)

        # ── 1. Build per-node K-view snapshots ──
        # Each node gets its own K windows based on ITS OWN activity
        node_view_edge_lists, node_view_centers = build_per_node_snapshots(
            batch.edge_index, batch.t, N,
            self.num_snapshots, mode=self.snapshot_mode
        )
        # node_view_centers: (N, K) — different per node!

        # ── 2. Encode each view with trust+similarity attention ──
        # Trust and similarity gates are applied INSIDE each view
        # State (x) is carried forward from view k to view k+1
        view_embs = self.view_encoder(
            x, node_view_edge_lists, node_view_centers,
            batch.edge_index, batch.t, batch.msg,
            self.memory, self.evo_bank, self.num_snapshots
        )
        # view_embs: (N, K, D)

        # ── 3. Integrate K views with per-node inter-view time deltas ──
        x_final = self.view_integrator(view_embs, node_view_centers)
        x_final = self.final_norm(x_final)

        # ── 4. Update memory and evolution bank ──
        t_per_node = torch.zeros(N, device=device)
        t_per_node.scatter_reduce_(
            0, batch.edge_index[0], batch.t.float(),
            reduce='amax', include_self=True
        )
        self.memory.write(torch.arange(N, device=device), x_final, t_per_node)

        src, dst = batch.edge_index
        neigh = torch.zeros(N, self.hidden_dim, device=device)
        cnt   = torch.zeros(N, 1, device=device)
        neigh.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.hidden_dim), x_final[src])
        cnt.scatter_add_(0, dst.unsqueeze(1), torch.ones(src.size(0), 1, device=device))
        neigh   = neigh / (cnt + 1e-6)
        unique_dst = dst.unique()
        self.evo_bank.write(unique_dst, neigh[unique_dst])

        return x_final


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Contrastive Projection Head
# ═══════════════════════════════════════════════════════════════════════════════

class ContrastiveHead(nn.Module):
    def __init__(self, hidden_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class FullModel(nn.Module):
    def __init__(self, num_nodes, in_dim, edge_dim, hidden_dim,
                 proj_dim=128, **gnn_kwargs):
        super().__init__()
        self.gnn  = MultiViewTemporalGNN(
            num_nodes, in_dim, edge_dim, hidden_dim, **gnn_kwargs
        )
        self.head = ContrastiveHead(hidden_dim, proj_dim)

    def forward(self, batch):
        return self.gnn(batch)

    def project(self, batch):
        h = self.gnn(batch)
        return h, self.head(h)