import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax

# ------------------------------------------------------------
# Low-pass extractor (unchanged)
# ------------------------------------------------------------
from layers import GraphConvolution

class LowPassExtractor(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.bn1 = nn.BatchNorm1d(2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.bn1(self.gc1(x, adj)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.bn2(self.gc2(x, adj)))
        return x


# ------------------------------------------------------------
# Encoder (FIX B: cached=False + correct NaN guards)
# ------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_motif, activation, learned_motif_weight=False):
        super().__init__()
        # Linear projection instead of GCN — avoids double smoothing with LowPassExtractor
        self.proj = nn.Linear(in_channels, 2 * out_channels)
        self.conv = GCNConv(2 * out_channels, out_channels, add_self_loops=False)

        self.motif = nn.Embedding(num_motif, 2 * out_channels)
        self.activation = activation

        # Ablation: replace the fixed tf*idf edge weight with a trained
        # per-motif-type scalar (edge_weight_motif is passed in as tf-only
        # in this mode; this parameter substitutes for idf, learned end-to-end
        # instead of hand-derived from document frequency).
        self.learned_motif_weight = learned_motif_weight
        if learned_motif_weight:
            self.motif_edge_scale = nn.Parameter(torch.ones(num_motif))

    def forward(
            self,
            x: torch.Tensor,
            edge_index_motif: torch.Tensor,
            edge_weight_motif: torch.Tensor,
            motif_num: int,
            motif_used: torch.Tensor,
        ):
            device = x.device
            N = x.size(0)

            if self.learned_motif_weight:
                src, dst = edge_index_motif[0], edge_index_motif[1]
                is_motif_edge = (src >= N) ^ (dst >= N)
                motif_id_per_edge = torch.where(src >= N, src - N, dst - N).clamp(0, motif_num - 1)
                scale = torch.ones_like(edge_weight_motif)
                scale[is_motif_edge] = self.motif_edge_scale[motif_id_per_edge[is_motif_edge]]
                edge_weight_motif = edge_weight_motif * scale

            # --------------------------------------------------
            # 1. Linear projection (no graph smoothing here — LowPassExtractor handles that)
            # --------------------------------------------------
            x = self.activation(self.proj(x))
            x = F.normalize(x, dim=1, eps=1e-12)

            if not torch.isfinite(x).all():
                raise RuntimeError("NaN/Inf in node features after projection")

            # --------------------------------------------------
            # 2. Motif embeddings (SAFE)
            # --------------------------------------------------
            motif_ids = torch.arange(motif_num, device=device)
            motif_emb = self.motif(motif_ids)

            # 🔑 HARD MASK UNUSED MOTIFS
            motif_emb = motif_emb * motif_used.unsqueeze(1)

            # Replace any numerical garbage
            motif_emb = torch.nan_to_num(
                motif_emb,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            # Normalize ONLY non-zero rows
            norm = motif_emb.norm(dim=1, keepdim=True)
            motif_emb = motif_emb / (norm + 1e-12)
            #motif_emb = F.dropout(motif_emb, p=0.6, training=self.training)

            # --------------------------------------------------
            # 3. Concatenate
            # --------------------------------------------------
            x_new = torch.cat([x, motif_emb], dim=0)

            if not torch.isfinite(x_new).all():
                raise RuntimeError("NaN/Inf in x_new BEFORE motif GCN")

            # --------------------------------------------------
            # 4. Motif-aware GCN
            # --------------------------------------------------
            x_new = self.conv(
                x_new,
                edge_index_motif,
                edge_weight=edge_weight_motif
            )
            x_new = self.activation(x_new)
            #x_new = F.dropout(x_new, p=0.6, training=self.training) 
            x_new = torch.nan_to_num(x_new, nan=0.0, posinf=0.0, neginf=0.0)

            # Keep node part
            x = x_new[:N]
            x = F.normalize(x, dim=1, eps=1e-12)

            if not torch.isfinite(x).all():
                raise RuntimeError("NaN/Inf in Encoder output")

            return x


# ------------------------------------------------------------
# Gating network (unchanged except safety clamp)
# ------------------------------------------------------------
class Gating(nn.Module):
    def __init__(self, nfeat, nhid, num_expert=2, dropout=0.5, temperature=1.0):
        super().__init__()
        self.fc1 = nn.Linear(2 * nfeat + nfeat + 1, nhid)
        self.norm1 = nn.LayerNorm(nhid)
        self.fc2 = nn.Linear(nhid, num_expert)
        self.dropout = dropout
        self.temperature = temperature
        self._x_cat = None
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, adj):
        if self._x_cat is None:
            AX = torch.spmm(adj, x)
            delta = AX - x

            deg = torch.sparse.sum(adj, dim=1).to_dense().unsqueeze(1)
            x_mean = AX / (deg + 1e-8)
            AX2 = torch.spmm(adj, x * x)
            var = AX2 / (deg + 1e-8) - x_mean * x_mean
            x_std = torch.sqrt(torch.clamp(var, min=0.0))

            self._x_cat = torch.cat([x, delta, x_std, deg], dim=1)

        h = F.relu(self.norm1(self.fc1(self._x_cat)))
        h = F.dropout(h, self.dropout, training=self.training)
        logits = self.fc2(h)
        return F.softmax(logits / self.temperature, dim=1)


# ------------------------------------------------------------
# Experts
# ------------------------------------------------------------
class LowPassExpert(nn.Module):
    def forward(self, low_feature, adj):
        return low_feature


# ------------------------------------------------------------
# FIX A: numerically stable sparse attention
# ------------------------------------------------------------
class SparseAttentionHighPass(nn.Module):
    def __init__(self, nhid, dropout=0.5, alpha=0.2):
        super().__init__()
        self.q = nn.Linear(nhid, nhid, bias=False)
        self.k = nn.Linear(nhid, nhid, bias=False)
        self.v = nn.Linear(nhid, nhid, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = dropout
        self.nhid = nhid

    def forward(self, residual, adj):
        Q = self.q(residual)
        K = self.k(residual)
        V = self.v(residual)

        edge_index = adj.coalesce().indices()
        src, dst = edge_index[0], edge_index[1]

        scores = (Q[src] * K[dst]).sum(dim=1) / (self.nhid ** 0.5)
        scores = self.leakyrelu(scores)

        # ✅ stable grouped softmax
        attn = softmax(scores, src)
        attn = F.dropout(attn, self.dropout, training=self.training)

        out = torch_scatter.scatter_add(
            attn.unsqueeze(1) * V[dst],
            src,
            dim=0,
            dim_size=residual.size(0)
        )
        return out


class HighPassExpert(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super().__init__()
        self.x_proj = nn.Linear(nfeat, nhid)
        self.attn = SparseAttentionHighPass(nhid, dropout)
        self.norm = nn.LayerNorm(nhid)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, low_feature, adj):
        x_proj = self.x_proj(x)
        residual = self.scale * (x_proj - low_feature)
        residual = self.norm(residual)
        return self.attn(residual, adj)


# ------------------------------------------------------------
# MOE model
# ------------------------------------------------------------
class MOE(nn.Module):
    def __init__(self, nfeat, nhid, motif_num, dropout=0.5, temperature=2.0, init_motif_gate=-3.0, use_motif=True, learned_motif_weight=False):
        super().__init__()
        self.use_motif = use_motif
        # Original pipeline unchanged — operates on raw features
        self.lowpass = LowPassExtractor(nfeat, nhid, dropout)
        self.gating = Gating(nfeat, nhid, dropout=dropout, temperature=temperature)
        self.experts = nn.ModuleList([
            LowPassExpert(),
            HighPassExpert(nfeat, nhid, dropout)
        ])

        # Motif encoder as a side branch — outputs nhid to match emb dim
        self.encoder = Encoder(nfeat, nhid, motif_num, activation=F.relu, learned_motif_weight=learned_motif_weight)

        # Learned gate for motif residual — dataset-specific init
        self.motif_gate = nn.Parameter(torch.tensor(init_motif_gate))

        self.lambda_corr = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, adj):
        low = self.lowpass(x, adj)
        gates = self.gating(x, adj)

        out = 0
        high = None
        for i, expert in enumerate(self.experts):
            if i == 0:
                e = expert(low, adj)
            else:
                e = expert(x, low, adj)
                high = e
            out = out + gates[:, i: i + 1] * e

        return out, gates, low, high

    def forward_with_correction(
        self, x, adj,
        support_idx, query_idx,
        n_way, k_shot, bandwidth,
        edge_index_motif, edge_weight_motif, motif_num, motif_used
    ):
        # Original pipeline on raw features (preserves baseline behavior)
        emb, gates, low, high = self.forward(x, adj)

        # Motif side branch — trained end-to-end with the main pipeline
        motif_feat = self.encoder(x, edge_index_motif, edge_weight_motif, motif_num, motif_used)
        gate_m = torch.sigmoid(self.motif_gate)
        emb = emb + gate_m * motif_feat

        z_dim = emb.size(1)
        support = emb[support_idx].view(n_way, k_shot, z_dim)
        proto = support.mean(dim=1)
        query = emb[query_idx]

        corrected = []
        lam = 0.5 * (torch.tanh(self.lambda_corr) + 1)

        for i in range(n_way):
            diff = query - proto[i]
            dist = (diff ** 2).sum(dim=1)
            logits = -dist / (2 * bandwidth ** 2)
            logits = torch.clamp(logits, -50, 0)
            w = torch.exp(logits)
            w = w / (w.sum() + 1e-8)
            corr = (w.unsqueeze(1) * diff).sum(dim=0)
            corrected.append(proto[i] + lam * corr)

        return emb, gates, low, high, torch.stack(corrected)
