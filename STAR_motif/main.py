import argparse
import os
import os.path as osp
import random
from time import perf_counter as t
import yaml
import json
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_edge, add_self_loops
from torch_geometric.nn import GCNConv, SGConv

from collections import defaultdict
from model import Encoder, MotifEncoder, Model, drop_feature, Decoder
from validation_concatenation import fs_dev, fs_test

from data_split import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

import numpy as np
from copy import deepcopy
from args import get_args
import networkx as nx
from sklearn.cluster import KMeans
import scipy.sparse as sp
from multiprocessing import Pool
import multiprocessing as mp


# ---------------------------------------------------------------------------
# Motif helpers (GRACE-style)
# ---------------------------------------------------------------------------

def get_n_hop_subgraph_sparse(adj, i):
    """Return 1-hop neighbourhood of node i using sparse adjacency."""
    row, col = adj.indices()
    mask = (row == i) | (col == i)
    neighbors = torch.unique(torch.cat([row[mask], col[mask]]))
    idx_total = torch.unique(torch.cat([neighbors, torch.tensor([i], device=adj.device)]))
    original_new_node = {n.item(): idx for idx, n in enumerate(idx_total)}
    new_original_node = {idx: n.item() for idx, n in enumerate(idx_total)}
    return idx_total, original_new_node, new_original_node


def get_3_node_subgraph_undirected(G, motif, o_n, n_o, node_num, max_node, seen, motif_adj, motif_x):
    cur = node_num
    for i in G.neighbors(cur):
        if i == cur:
            continue
        for j in G.neighbors(i):
            if j == i or j == cur:
                continue
            tri = tuple(sorted([cur, i, j]))
            if tri in seen:
                continue
            seen.add(tri)

            subgraph = G.subgraph([cur, i, j])
            h = nx.weisfeiler_lehman_graph_hash(subgraph, node_attr="Labels")

            if h not in motif:
                motif[h] = np.zeros(max_node, dtype=np.int32)
                motif_adj[h] = []
                motif_x[h] = []

            motif[h][i] += 1
            motif[h][cur] += 1
            motif[h][j] += 1

            cur_adj = nx.to_numpy_array(subgraph)
            adj_tensor = torch.tensor(cur_adj, dtype=torch.float32)
            motif_adj[h].append(adj_tensor)
            motif_x[h].append(adj_tensor.sum(dim=1))
    return motif


def save_motifs_txt(motif_dict, filepath):
    """Serialize motif count vectors to a text file for caching."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w") as f:
        for motif_id, (_, vec) in enumerate(motif_dict.items()):
            nz = np.nonzero(vec)[0]
            for node_id in nz:
                f.write(f"{motif_id} {node_id} {int(vec[node_id])}\n")


def load_motifs_txt(filepath, num_original_nodes):
    """Load motif count vectors from a cached text file."""
    motif = {}
    with open(filepath, "r") as f:
        for line in f:
            motif_id, node_id, count = map(int, line.strip().split())
            if motif_id not in motif:
                motif[motif_id] = np.zeros(num_original_nodes, dtype=np.int32)
            motif[motif_id][node_id] = count
    return motif


def _motif_hash_3node(a, b, c, labels, adj_set):
    """WL hash for a 3-node motif without building a full subgraph object."""
    la, lb, lc = labels[a], labels[b], labels[c]
    e_ab = int((min(a,b), max(a,b)) in adj_set)
    e_bc = int((min(b,c), max(b,c)) in adj_set)
    e_ac = int((min(a,c), max(a,c)) in adj_set)
    # Build a tiny 3-node graph with remapped node ids
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2])
    nx.set_node_attributes(g, {0: la, 1: lb, 2: lc}, name="Labels")
    if e_ab: g.add_edge(0, 1)
    if e_bc: g.add_edge(1, 2)
    if e_ac: g.add_edge(0, 2)
    return nx.weisfeiler_lehman_graph_hash(g, node_attr="Labels")


def _motif_worker(args_tuple):
    """Worker: extract motifs for a chunk of nodes, return partial motif dict."""
    node_ids, edge_list, node_labels, original_node_num = args_tuple
    # Build adjacency list (dict of sets) — faster than NetworkX for neighbor lookup
    adj_list = {i: set() for i in range(original_node_num)}
    adj_set = set()
    for u, v in edge_list:
        adj_list[u].add(v)
        adj_list[v].add(u)
        adj_set.add((min(u, v), max(u, v)))

    motif_3 = {}
    seen = set()
    for i in node_ids:
        for nb_i in adj_list[i]:
            if nb_i == i:
                continue
            for nb_j in adj_list[nb_i]:
                if nb_j == nb_i or nb_j == i:
                    continue
                tri = tuple(sorted([i, nb_i, nb_j]))
                if tri in seen:
                    continue
                seen.add(tri)
                h = _motif_hash_3node(i, nb_i, nb_j, node_labels, adj_set)
                if h not in motif_3:
                    motif_3[h] = np.zeros(original_node_num, dtype=np.int32)
                motif_3[h][i] += 1
                motif_3[h][nb_i] += 1
                motif_3[h][nb_j] += 1
    return motif_3


def merge_motif_dicts(dicts):
    """Merge list of partial motif dicts by summing counts."""
    merged = {}
    for d in dicts:
        for h, vec in d.items():
            if h not in merged:
                merged[h] = vec.copy()
            else:
                merged[h] += vec
    return merged


def add_edge_with_tf_idf(Graph, node, motif_vec, total_num):
    """Add TF-IDF weighted edges between original nodes and a motif node."""
    idx = np.nonzero(motif_vec)[0]
    motif_len = len(idx)
    if motif_len == 0:
        return Graph
    idf = np.log(total_num / motif_len)
    for i in idx:
        a = motif_vec[i]
        tf = np.log(a)
        w = tf * idf
        if w > 0:
            Graph.add_edge(int(i), int(node), weight=float(w))
    return Graph


def build_edge_index_weight_from_nx(G, num_nodes_total, device, min_weight=1e-3, self_loop_weight=1.0):
    """Convert a NetworkX graph (with 'weight' attributes) to edge_index + edge_weight tensors."""
    edge_index = []
    edge_weight = []
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        if not (w > 0 and np.isfinite(w)):
            continue
        w = max(w, min_weight)
        edge_index.append([u, v])
        edge_index.append([v, u])
        edge_weight.append(w)
        edge_weight.append(w)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value=self_loop_weight, num_nodes=num_nodes_total
    )

    # Sanity check: no zero-degree nodes
    deg = torch.zeros(num_nodes_total)
    deg.index_add_(0, edge_index[0], edge_weight.cpu())
    if deg.min().item() <= 0:
        raise RuntimeError("Zero-degree node in motif graph even after self-loops")

    return edge_index.to(device), edge_weight.to(device)


def compute_motif_used(edge_index_motif, original_node_num, motif_num, device):
    """Return a boolean mask of motif nodes that have at least one edge."""
    nodes = edge_index_motif.flatten()
    motif_ids = nodes - original_node_num
    motif_ids = motif_ids[(motif_ids >= 0) & (motif_ids < motif_num)]
    used = torch.zeros(motif_num, dtype=torch.bool, device=device)
    used[motif_ids] = True
    return used


# ---------------------------------------------------------------------------
# Relabeling (unchanged)
# ---------------------------------------------------------------------------

def relabeling(labels, train_class, dev_class, test_class, id_by_class):
    print("Start relabeling...")
    labels = labels.tolist()
    contrast_labels = deepcopy(labels)
    masked_class = dev_class + test_class
    masked_idx = []
    for cla in masked_class:
        masked_idx.extend(id_by_class[cla])

    train_class.sort()
    train_class_map = {i: train_class.index(i) for i in train_class}

    tmp_class = len(train_class)
    for cla, idx_list in id_by_class.items():
        if cla in train_class:
            for idx in idx_list:
                contrast_labels[idx] = train_class_map[cla]
        else:
            for idx in idx_list:
                contrast_labels[idx] = tmp_class
                tmp_class += 1
    print("Relabeling finished!")
    return contrast_labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model: Model, x, contrast_labels, edge_index,
          edge_index_motif, edge_weight_motif, motif_num, motif_used,
          optimizer):
    model.train()
    optimizer.zero_grad()

    edge_index_1 = dropout_edge(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_edge(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)

    z1 = model(x_1, edge_index_1, edge_index_motif, edge_weight_motif, motif_num, motif_used)
    z2 = model(x_2, edge_index_2, edge_index_motif, edge_weight_motif, motif_num, motif_used)

    device = z1.device

    z1_d, z2_d = z1.detach(), z2.detach()

    # Chunked topk on GPU to avoid N×N OOM
    chunk = 512
    topk_idx_list = []
    N = z1_d.size(0)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        sim_chunk = z1_d[start:end] @ z2_d.T
        topk_idx_list.append(torch.topk(sim_chunk, k=args.topk, dim=1, largest=True).indices)
    topnk_idx = torch.cat(topk_idx_list, dim=0)
    topk_idx = topnk_idx[:, torch.randperm(topnk_idx.size(1), device=device)[:args.topk]]
    set_emb = z2_d[topk_idx]

    set_1, set_2 = torch.split(set_emb, args.topk // 2, dim=1)

    zs_1, zs_2 = model.set_forward(set_1), model.set_forward(set_2)

    if not args.sup:
        # Subsample nodes for contrastive loss on large graphs to reduce matrix size
        if args.contrast_node_subsample > 0 and z1.size(0) > args.contrast_node_subsample:
            idx = torch.randperm(z1.size(0), device=device)[:args.contrast_node_subsample]
            ins_loss = model.loss(z1[idx], z2[idx], batch_size=args.contrast_batch_size)
            set_loss = model.loss(zs_1[idx], zs_2[idx], batch_size=args.contrast_batch_size)
        else:
            ins_loss = model.loss(z1, z2, batch_size=args.contrast_batch_size)
            set_loss = model.loss(zs_1, zs_2, batch_size=args.contrast_batch_size)
        loss = alpha * ins_loss + beta * set_loss

    loss.backward()

    # Sanitize gradients and clip (GRACE-style robustness)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.nan_to_num_(0.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Main train/eval loop
# ---------------------------------------------------------------------------

def train_eval():
    dataset, train_idx, id_by_class, train_class, dev_class, test_class = split(args.dataset)
    if args.dataset in ['ogbn-arxiv', 'CoraFull', 'coauthor-cs', 'Amazon-Computer',
                        'Reddit', 'Cora', 'WikiCS', 'CiteSeer', 'Cora_ML']:
        data = dataset[0]
    elif args.dataset in ['dblp']:
        data = dataset.data
    else:
        data = dataset

    original_node_num = data.x.shape[0]

    # Build sparse adjacency (for neighbour lookup)
    adj_sparse = torch.sparse_coo_tensor(
        indices=data.edge_index,
        values=torch.ones(data.edge_index.size(1)),
        size=(original_node_num, original_node_num)
    ).coalesce()

    # Build NetworkX graph (no dense materialisation for large graphs)
    edge_index_np = data.edge_index.cpu().numpy().T
    G = nx.Graph()
    G.add_nodes_from(range(original_node_num))
    G.add_edges_from(edge_index_np)

    # Node labels for WL hash (KMeans on normalised features)
    feat_np = normalize(data.x.numpy(), axis=1)
    n_clusters = 4 if args.way == 1 else kmeans_clusters.get(args.dataset, 5)
    temp_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(feat_np)
    nx.set_node_attributes(G, {i: int(temp_labels[i]) for i in range(len(temp_labels))}, name="Labels")

    # -------------------------------------------------------------------
    # Motif extraction / loading from cache
    # -------------------------------------------------------------------
    os.makedirs("motif_cache", exist_ok=True)
    os.makedirs("savepoint", exist_ok=True)
    motif_file = f"motif_cache/motifs_3node_{n_clusters}_{args.dataset}.txt"
    if not os.path.exists(motif_file):
        print("Extracting motifs...")
        edge_list = list(G.edges())
        node_labels = nx.get_node_attributes(G, "Labels")
        # Build dict-of-sets adjacency — faster than NetworkX neighbor iteration
        adj_list = {i: set() for i in range(original_node_num)}
        adj_set = set()
        for u, v in edge_list:
            adj_list[int(u)].add(int(v))
            adj_list[int(v)].add(int(u))
            adj_set.add((min(int(u), int(v)), max(int(u), int(v))))
        motif_3 = {}
        seen = set()
        for i in range(original_node_num):
            if i % 10000 == 0:
                print(f"  motif extraction: {i}/{original_node_num}")
            for nb_i in adj_list[i]:
                for nb_j in adj_list[nb_i]:
                    if nb_j == nb_i or nb_j == i:
                        continue
                    tri = (min(i, nb_i, nb_j),
                           sorted([i, nb_i, nb_j])[1],
                           max(i, nb_i, nb_j))
                    if tri in seen:
                        continue
                    seen.add(tri)
                    h = _motif_hash_3node(i, nb_i, nb_j, node_labels, adj_set)
                    if h not in motif_3:
                        motif_3[h] = np.zeros(original_node_num, dtype=np.int32)
                    motif_3[h][i] += 1
                    motif_3[h][nb_i] += 1
                    motif_3[h][nb_j] += 1
        save_motifs_txt(motif_3, motif_file)
        print(f"Motifs saved: {motif_file}")
    else:
        print(f"Loading motifs from cache: {motif_file}")
        motif_3 = load_motifs_txt(motif_file, original_node_num)

    # -------------------------------------------------------------------
    # Build motif-augmented graph with TF-IDF edges
    # -------------------------------------------------------------------
    motif_keys = list(motif_3.keys())
    motif_num = len(motif_keys)
    print(f"Number of motif types: {motif_num}")

    for mi, k in enumerate(motif_keys):
        mnode = original_node_num + mi
        G.add_node(mnode)
        G = add_edge_with_tf_idf(G, mnode, motif_3[k], original_node_num)

    # Global min-max normalisation to [0, 0.5]
    motif_weights = [
        d["weight"]
        for u, v, d in G.edges(data=True)
        if "weight" in d and (u >= original_node_num or v >= original_node_num)
    ]
    if len(motif_weights) == 0:
        raise RuntimeError("No motif edges with weights found")

    min_w, max_w = min(motif_weights), max(motif_weights)
    scale, eps = 0.5, 1e-8
    for u, v, d in G.edges(data=True):
        if "weight" in d and (u >= original_node_num or v >= original_node_num):
            d["weight"] = float((d["weight"] - min_w) / (max_w - min_w + eps) * scale)

    # Prune weakest 50% of motif edges
    new_weights = [
        d["weight"]
        for u, v, d in G.edges(data=True)
        if "weight" in d and (u >= original_node_num or v >= original_node_num)
    ]
    thr = np.percentile(new_weights, 50)
    edges_to_remove = [
        (u, v) for u, v, d in G.edges(data=True)
        if (u >= original_node_num or v >= original_node_num) and d.get("weight", 0.0) < thr
    ]
    G.remove_edges_from(edges_to_remove)

    # Per-node degree cap: keep at most MAX_NODE_MOTIF_DEG motif connections per node
    MAX_NODE_MOTIF_DEG = 20
    node_edge_map = {}
    for u, v, d in G.edges(data=True):
        if u < original_node_num and v >= original_node_num:
            node_edge_map.setdefault(u, []).append((u, v, d))
        elif v < original_node_num and u >= original_node_num:
            node_edge_map.setdefault(v, []).append((u, v, d))

    edges_to_remove = []
    for node, edges in node_edge_map.items():
        if len(edges) > MAX_NODE_MOTIF_DEG:
            edges_sorted = sorted(edges, key=lambda e: e[2].get("weight", 0), reverse=True)
            edges_to_remove.extend([(u, v) for u, v, _ in edges_sorted[MAX_NODE_MOTIF_DEG:]])
    G.remove_edges_from(edges_to_remove)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    num_nodes_total = original_node_num + motif_num
    edge_index_motif, edge_weight_motif = build_edge_index_weight_from_nx(
        G, num_nodes_total=num_nodes_total, device=device, min_weight=1e-3, self_loop_weight=1.0
    )

    motif_used = compute_motif_used(edge_index_motif, original_node_num, motif_num, device)
    print(f"motif_used: {motif_used.sum().item()} / {motif_num}")

    # -------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------
    print(f"Building model (hidden={num_hidden})")
    in_features = data.x.shape[1]
    encoder = Encoder(in_features, num_hidden, activation)
    motif_encoder = MotifEncoder(num_hidden, motif_num, activation)
    decoder = Decoder(set_name, num_hidden, num_heads)
    init_gate = config.get('init_motif_gate', args.init_motif_gate)
    model = Model(encoder, motif_encoder, decoder, num_hidden, num_proj_hidden, tau,
                  init_motif_gate=init_gate).to(device)

    if hasattr(data, 'to'):
        data = data.to(device)
    else:
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.y = data.y.to(device)
    contrast_labels = relabeling(data.y, train_class, dev_class, test_class, id_by_class)
    contrast_labels = torch.LongTensor(contrast_labels).to(device)

    # Three param groups: motif embeddings (low LR), rest of encoder, rest of model
    motif_params, encoder_params, other_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder.motif" in name:
            motif_params.append(param)
        elif name.startswith("encoder"):
            encoder_params.append(param)
        else:
            other_params.append(param)

    enc_lr_ratio = encoder_lr_ratio_map.get(args.dataset, 0.1)
    optimizer = torch.optim.Adam([
        {"params": motif_params,   "lr": learning_rate * enc_lr_ratio, "weight_decay": weight_decay},
        {"params": encoder_params, "lr": learning_rate * enc_lr_ratio, "weight_decay": weight_decay},
        {"params": other_params,   "lr": learning_rate,                "weight_decay": weight_decay},
    ])

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    cnt_wait = 0
    best_acc = 0
    for epoch in range(1, num_epochs + 1):
        _ = train(model, data.x, contrast_labels, data.edge_index,
                  edge_index_motif, edge_weight_motif, motif_num, motif_used,
                  optimizer)

        if (epoch - 1) % 10 == 0:
            final_mean, final_std, _ = fs_dev(
                model, data.x, data.edge_index, edge_index_motif, edge_weight_motif,
                motif_num, motif_used, data.y,
                setting['test_num'], id_by_class, dev_class,
                setting['n_way'], setting['k_shot'], setting['m_qry']
            )
            print("=" * 80)
            print(f"epoch={epoch}  novel_dev_acc={final_mean:.4f}  std={final_std:.4f}")
            if best_acc < final_mean:
                best_acc = final_mean
                cnt_wait = 0
                torch.save(model.state_dict(), f'./savepoint/{args.dataset}_model.pkl')
            else:
                cnt_wait += 1

        if cnt_wait == setting['patience']:
            print('Early stopping!')
            break

    print("=== Final Test ===")
    model.load_state_dict(torch.load(f'./savepoint/{args.dataset}_model.pkl'))
    print("Model loaded.")
    final_mean, final_std, final_interval = fs_test(
        model, data.x, data.edge_index, edge_index_motif, edge_weight_motif,
        motif_num, motif_used, data.y, train_idx,
        setting['test_num'], id_by_class, test_class,
        setting['n_way'], setting['k_shot'], setting['m_qry'], device, args
    )
    print(f"novel_test_acc={final_mean:.4f}  std={final_std:.4f}  interval={final_interval:.4f}")
    return final_mean, final_std, final_interval


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Dataset-specific hyperparameters (from GRACE utils.py)
kmeans_clusters = {
    'CoraFull': 15, 'Cora': 6, 'CiteSeer': 10, 'Cora_ML': 5,
    'Amazon-Computer': 10, 'coauthor-cs': 15, 'WikiCS': 10, 'ogbn-arxiv': 25, 'dblp': 15,
}
encoder_lr_ratio_map = {
    'CoraFull': 0.1, 'Cora': 0.2, 'CiteSeer': 0.1, 'Cora_ML': 0.1,
    'Amazon-Computer': 0.1, 'coauthor-cs': 0.1, 'WikiCS': 0.1, 'ogbn-arxiv': 0.1,
}
gate_clamp_map = {
    'CoraFull': 0.2, 'Cora': 0.2, 'CiteSeer': 0.01, 'Cora_ML': 0.2,
    'Amazon-Computer': 0.2, 'coauthor-cs': 0.2, 'WikiCS': 0.2, 'ogbn-arxiv': 0.2,
}
topk_map = {
    'Cora': 20, 'Cora_ML': 20,
    'CiteSeer': 10,
    'coauthor-cs': 10, 'ogbn-arxiv': 10, 'CoraFull': 10, 'WikiCS': 10,
    'Amazon-Computer': 10,
}
# Datasets too large for full N×N contrastive matrix — use batched loss
contrastive_batch_size_map = {
    'coauthor-cs': 1000, 'ogbn-arxiv': 1000, 'CoraFull': 1000, 'WikiCS': 1000,
}
# Subsample this many nodes per epoch for contrastive loss (0 = use all)
contrast_node_subsample_map = {
    'coauthor-cs': 4000, 'ogbn-arxiv': 16000, 'CoraFull': 4000, 'dblp': 4000,
}

if __name__ == '__main__':
    args = get_args()
    args.topk = topk_map.get(args.dataset, 20)
    args.contrast_batch_size = contrastive_batch_size_map.get(args.dataset, 0)
    args.contrast_node_subsample = contrast_node_subsample_map.get(args.dataset, 0)
    print(args)
    setting = {
        'n_way': args.way, 'k_shot': args.shot, 'm_qry': args.query,
        'test_num': args.test_num, 'patience': args.patience,
    }
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    base_seed = config['seed']
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    learning_rate = args.lr if args.lr is not None else config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = args.drop_edge_rate_1
    drop_edge_rate_2 = args.drop_edge_rate_2
    drop_feature_rate_1 = args.drop_feature_rate_1
    drop_feature_rate_2 = args.drop_feature_rate_2
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    set_name = args.set_encoder
    num_heads = args.num_heads
    alpha = args.alpha
    beta = args.beta

    acc_mean = []
    acc_std = []
    acc_interval = []

    for turn in range(1, args.turns + 1):
        seed = base_seed + turn - 1
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"\n{'='*60}")
        print(f"Turn {turn} / {args.turns}")
        print(f"{'='*60}")
        m, s, iv = train_eval()
        acc_mean.append(m)
        acc_std.append(s)
        acc_interval.append(iv)
        print(f"Turn {turn} (seed={seed}) result — acc: {m:.4f}  std: {s:.4f}  interval: {iv:.4f}")

    print(f"\n{'='*60}")
    print(f"{'='*60}")
    print(f"{args.turns}-turn average — acc: {np.mean(acc_mean):.4f}  std: {np.std(acc_mean):.4f}  interval: {np.mean(acc_interval):.4f}")
    print(f"{'='*60}")

    result = defaultdict(list)
    result[tuple([np.mean(acc_mean), np.std(acc_mean)])] = {
        'way': args.way,
        'shot': args.shot,
        'drop_feature_rate_1': args.drop_feature_rate_1,
        'drop_feature_rate_2': args.drop_feature_rate_2,
        'temperature': args.temperature,
        'beta': args.beta,
        'gamma': args.gamma,
        'alpha': args.alpha,
        'epoch': args.epoch,
        'weight_decay': args.wd,
    }
    os.makedirs("./result", exist_ok=True)
    with open(f"./result/{args.dataset}_res.txt", "a+") as f:
        f.write(json.dumps({str(k): result[k] for k in result}, indent=4))
