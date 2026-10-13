import argparse, os, time, math, random, warnings, json
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import networkx as nx

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from torch_geometric.utils import add_self_loops

from data_split import *
from utils import *
from models import MOE  # make sure this is your updated MOE

warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------
# args
# ----------------------------
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dataset", default="CiteSeer")
    parser.add_argument("--n_way", type=int, default=2)
    parser.add_argument("--k_shot", type=int, default=3)
    parser.add_argument("--m_qry", type=int, default=10)
    parser.add_argument("--test_num", type=int, default=100)
    parser.add_argument("--loss_temperature", type=float, default=0.5)
    parser.add_argument("--loss_cls", type=float, default=1.0)
    parser.add_argument("--loss_orth", type=float, default=0.2)
    parser.add_argument("--model_temperature", type=float, default=2.0)
    parser.add_argument("--stepsize", type=int, default=10)
    parser.add_argument("--stepgamma", type=float, default=0.99)
    parser.add_argument("--tfidf_percentile", type=int, default=50)
    parser.add_argument("--max_node_motif_deg", type=int, default=20)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--inductive", action="store_true",
                         help="Remove edges that cross train/dev/test class splits before "
                              "building the base graph and motifs (inductive setting).")
    return parser


def remove_cross_split_edges(edge_index, num_nodes, id_by_class, train_class, dev_class, test_class):
    """Drop every edge whose endpoints fall in different class-splits.

    Standard episodic few-shot node classification (as implemented in `split()`)
    keeps the full graph and only partitions *labels* into train/dev/test — the
    encoder can still message-pass across a train/test boundary, which is a
    transductive assumption. This filters edge_index down to edges that are
    fully inside one split, so train/dev/test subgraphs are structurally
    disjoint (inductive setting).
    """
    node_split = torch.full((num_nodes,), -1, dtype=torch.long)
    for cla in train_class:
        node_split[id_by_class[cla]] = 0
    for cla in dev_class:
        node_split[id_by_class[cla]] = 1
    for cla in test_class:
        node_split[id_by_class[cla]] = 2

    src, dst = edge_index
    keep = (node_split[src] == node_split[dst]) & (node_split[src] >= 0)
    n_before = edge_index.size(1)
    filtered = edge_index[:, keep]
    n_after = filtered.size(1)
    print(f"[inductive] cross-split edges removed: {n_before} -> {n_after} edges "
          f"({(1 - n_after / max(n_before, 1)) * 100:.1f}% dropped)")
    return filtered


# ----------------------------
# motif helpers
# ----------------------------
def get_n_hop_subgraph_sparse(adj, i):
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
    with open(filepath, "w") as f:
        for motif_id, (_, vec) in enumerate(motif_dict.items()):
            nz = np.nonzero(vec)[0]
            for node_id in nz:
                f.write(f"{motif_id} {node_id} {int(vec[node_id])}\n")


def load_motifs_txt(filepath, num_original_nodes):
    motif = {}
    with open(filepath, "r") as f:
        for line in f:
            motif_id, node_id, count = map(int, line.strip().split())
            if node_id >= num_original_nodes:
                continue
            if motif_id not in motif:
                motif[motif_id] = np.zeros(num_original_nodes, dtype=np.int32)
            motif[motif_id][node_id] = count
    return motif


def add_edge_with_tf_idf(Graph, node, motif_vec, total_num):
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

    # degree sanity
    deg = torch.zeros(num_nodes_total)
    deg.index_add_(0, edge_index[0], edge_weight.cpu())
    if deg.min().item() <= 0:
        raise RuntimeError("Zero-degree node in motif graph even after self-loops")

    return edge_index.to(device), edge_weight.to(device)


def compute_motif_used(edge_index_motif, original_node_num, motif_num, device):
    nodes = edge_index_motif.flatten()
    motif_ids = nodes - original_node_num
    motif_ids = motif_ids[(motif_ids >= 0) & (motif_ids < motif_num)]
    used = torch.zeros(motif_num, dtype=torch.bool, device=device)
    used[motif_ids] = True
    return used


# ----------------------------
# main
# ----------------------------
def main():
    args = get_parser().parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # per-dataset hyperparameters (utils.py)
    dropout = dropt.get(args.dataset, 0.0)
    hidden_size = hidden.get(args.dataset, 32)
    patience_val = patience.get(args.dataset, 10)
    lr_val = lr.get(args.dataset, 0.001)
    gate_val = gate.get(args.dataset, 0.1)
    contrast_val = contrast.get(args.dataset, 1.0)
    init_motif_gate = motif_gate_init.get(args.dataset, -3.0)
    enc_lr_ratio = encoder_lr_ratio.get(args.dataset, 0.1)
    n_clusters = kmeans_clusters.get(args.dataset, 5)

    # data
    data, train_idx, id_by_class, train_class, dev_class, test_class, degree_inv = split(args.dataset)
    original_node_num = data.x.size(0)

    if args.inductive:
        data.edge_index = remove_cross_split_edges(
            data.edge_index, original_node_num, id_by_class, train_class, dev_class, test_class
        )

    # build base sparse adj for your GraphConvolution + spmm
    adj = torch.sparse_coo_tensor(
        indices=data.edge_index,
        values=torch.ones(data.edge_index.size(1)),
        size=(original_node_num, original_node_num)
    ).coalesce()

    # build networkx graph from edge_index (no dense)
    edge_index_np = data.edge_index.cpu().numpy().T
    G = nx.Graph()
    G.add_nodes_from(range(original_node_num))   # <-- IMPORTANT: include isolated nodes
    G.add_edges_from(edge_index_np)

    suffix = "_inductive" if args.inductive else ""

    # node labels for WL hash: KMeans on normalized node features
    feat_np = normalize(data.x.cpu().numpy(), axis=1)
    temp_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(feat_np)
    nx.set_node_attributes(G, {i: int(temp_labels[i]) for i in range(len(temp_labels))}, name="Labels")

    # extract / load motifs
    motif_file = f"{BASE_DIR}/motifs_3node_{n_clusters}_{args.dataset}{suffix}.txt"
    motif_list_adj, motif_list_x = {}, {}
    seen = set()

    if not os.path.exists(motif_file):
        print("Extracting motifs...")
        motif_3 = {}
        for i in range(original_node_num):
            idx_total, o2n, n2o = get_n_hop_subgraph_sparse(adj, i)
            sub_G = G.subgraph(idx_total.cpu().tolist())
            motif_3 = get_3_node_subgraph_undirected(
                sub_G, motif_3, o2n, n2o, i, original_node_num, seen, motif_list_adj, motif_list_x
            )
        save_motifs_txt(motif_3, motif_file)
        print("Motifs saved:", motif_file)
    else:
        print("Loading motifs:", motif_file)
        motif_3 = load_motifs_txt(motif_file, original_node_num)

    # add motif nodes + TF-IDF edges
    # motif_3 is dict: hash->vec OR id->vec (after load); handle both
    motif_keys = list(motif_3.keys())
    motif_num = len(motif_keys)

    for mi, k in enumerate(motif_keys):
        mnode = original_node_num + mi
        G.add_node(mnode)
        vec = motif_3[k]
        G = add_edge_with_tf_idf(G, mnode, vec, original_node_num)

    # optional: rescale weights globally (safe)
    motif_edges = [
        d["weight"]
        for u, v, d in G.edges(data=True)
        if ("weight" in d) and (u >= original_node_num or v >= original_node_num)
    ]

    if len(motif_edges) == 0:
        raise RuntimeError("No motif edges with weights found")


    scale = 0.5
    eps = 1e-8
    weights = [
        d["weight"]
        for u, v, d in G.edges(data=True)
        if ("weight" in d) and (u >= original_node_num or v >= original_node_num)
    ]

    min_w = min(weights)
    max_w = max(weights)

    for u, v, d in G.edges(data=True):
        if ("weight" in d) and (u >= original_node_num or v >= original_node_num):
            d["weight"] = float(
                (d["weight"] - min_w) / (max_w - min_w + eps) * scale
            )

    new_w = [d["weight"] for u, v, d in G.edges(data=True) if ("weight" in d) and (u >= original_node_num or v >= original_node_num)]
    thr = np.percentile(new_w, args.tfidf_percentile)
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True)
                       if (u >= original_node_num or v >= original_node_num) and d.get("weight", 0.0) < thr]
    G.remove_edges_from(edges_to_remove)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes_total = original_node_num + motif_num

    edge_index_motif, edge_weight_motif = build_edge_index_weight_from_nx(
        G, num_nodes_total=num_nodes_total, device=device, min_weight=1e-3, self_loop_weight=1.0
    )
    MAX_NODE_MOTIF_DEG = args.max_node_motif_deg

    node_deg = {}
    edges_to_remove = []

    for u, v, d in G.edges(data=True):
        if u < original_node_num and v >= original_node_num:
            node = u
        elif v < original_node_num and u >= original_node_num:
            node = v
        else:
            continue

        node_deg.setdefault(node, []).append((u, v, d))

    # prune per node
    for node, edges in node_deg.items():
        if len(edges) > MAX_NODE_MOTIF_DEG:
            # keep strongest edges only
            edges = sorted(edges, key=lambda x: x[2].get("weight", 0), reverse=True)
            edges_to_remove.extend([(u, v) for u, v, _ in edges[MAX_NODE_MOTIF_DEG:]])

    G.remove_edges_from(edges_to_remove)

    deg = torch.zeros(num_nodes_total, device=edge_weight_motif.device)
    deg.index_add_(0, edge_index_motif[0], edge_weight_motif)
    node_deg = deg[:original_node_num]
    motif_deg = deg[original_node_num:]

    print("node motif-degree mean/max:", node_deg.mean().item(), node_deg.max().item())
    print("motif degree mean/max:", motif_deg.mean().item(), motif_deg.max().item())

    motif_used = compute_motif_used(edge_index_motif, original_node_num, motif_num, device)
    print("motif_used:", motif_used.sum().item(), "/", motif_num)


    # normalize base adj for your GraphConvolution-based parts (as in your code)
    adj_cpu = adj.to("cpu")
    I = torch.sparse_coo_tensor(
        indices=torch.stack([torch.arange(original_node_num), torch.arange(original_node_num)]),
        values=torch.ones(original_node_num),
        size=(original_node_num, original_node_num)
    )
    adj_cpu = (adj_cpu + I).coalesce()
    deg = torch.sparse.sum(adj_cpu, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[~torch.isfinite(deg_inv_sqrt)] = 0.0
    D = torch.sparse_coo_tensor(
        indices=torch.stack([torch.arange(original_node_num), torch.arange(original_node_num)]),
        values=deg_inv_sqrt,
        size=(original_node_num, original_node_num)
    )
    adj_norm = torch.sparse.mm(torch.sparse.mm(D, adj_cpu), D).coalesce()


    # move tensors
    features = data.x.to(device)
    labels = data.y.to(device)
    adj_norm = adj_norm.to(device)

    print("edge_index size:", edge_index_motif.size())
    print("num edges:", edge_index_motif.size(1))
    print("hidden dim:", features.size(1))

    # ----------------------------
    # train / test
    # ----------------------------
    def train(model, optimizer):
        model.train()
        optimizer.zero_grad()

        sampled_class = random.sample(range(len(train_class)), args.n_way)
        id_support, id_query = [], []
        for c in sampled_class:
            temp = random.sample(id_by_class[c], args.k_shot + args.m_qry)
            id_support.extend(temp[:args.k_shot])
            id_query.extend(temp[args.k_shot:])

        emb, gate_w, low_f, high_f, prot = model.forward_with_correction(
            features, adj_norm,
            id_support, id_query,
            args.n_way, args.k_shot, bandwidth=1.0,
            edge_index_motif=edge_index_motif,
            edge_weight_motif=edge_weight_motif,
            motif_num=motif_num,
            motif_used=motif_used
        )

        query_emb = emb[id_query]
        dists = euclidean_dist(query_emb, prot)
        output = F.log_softmax(-dists, dim=1)

        labels_new = torch.LongTensor([sampled_class.index(int(i)) for i in labels[id_query]]).to(device)
        labels_support = torch.LongTensor([sampled_class.index(int(i)) for i in labels[id_support]]).to(device)
        labels_query = torch.LongTensor([sampled_class.index(int(i)) for i in labels[id_query]]).to(device)

        loss_total = new_loss_function_with_contrast(
            output, labels_new, gate_w, low_f, high_f,
            emb[id_support], labels_support, emb[id_query], labels_query,
            lambda_cls=args.loss_cls, lambda_gate=gate_val,
            lambda_orth=args.loss_orth, lambda_contrast=contrast_val,
            temperature=args.loss_temperature
        )

        loss_total.backward()

        # ---- CRITICAL: sanitize + mask motif grads ----
        for p in model.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(0.0)

        # 2. clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        acc = accuracy(output.cpu().detach(), labels_new.cpu().detach())
        f1s = f1(output.cpu().detach(), labels_new.cpu().detach())
        return loss_total.item(), acc, f1s

    def test(model, eval_class):
        model.eval()
        accs = []
        for _ in range(args.test_num):
            sid, qid, cls_sel = test_task_generator(id_by_class, eval_class, args.n_way, args.k_shot, args.m_qry)
            emb, gate_w, low_f, high_f, prot = model.forward_with_correction(
                features, adj_norm,
                sid, qid,
                args.n_way, args.k_shot, 1.0,
                edge_index_motif,
                edge_weight_motif,
                motif_num,
                motif_used
            )
            dists = euclidean_dist(emb[qid], prot)
            out = F.log_softmax(-dists, dim=1)
            y = torch.LongTensor([cls_sel.index(int(i)) for i in labels[qid]]).to(device)
            accs.append(accuracy(out.cpu().detach(), y.cpu().detach()))
        m = float(np.mean(accs))
        s = float(np.std(accs))
        interval = 1.96 * (s / np.sqrt(len(accs)))
        return m, s, interval

    def train_eval():
        model = MOE(
            nfeat=features.size(1),
            nhid=hidden_size,
            dropout=dropout,
            temperature=args.model_temperature,
            motif_num=motif_num,
            init_motif_gate=init_motif_gate
        ).to(device)

        # sanity: make sure embedding matches motif_num
        assert model.encoder.motif.num_embeddings == motif_num


        encoder_params = []
        motif_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("encoder.motif"):
                motif_params.append(param)
            elif name.startswith("encoder"):
                encoder_params.append(param)
            else:
                other_params.append(param)

        optimizer = torch.optim.Adam(
            [
                {"params": motif_params, "lr": lr_val * enc_lr_ratio, "weight_decay": 5e-4},
                {"params": encoder_params, "lr": lr_val * enc_lr_ratio, "weight_decay": 5e-4},
                {"params": other_params, "lr": lr_val, "weight_decay": 5e-4},
            ],
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.stepgamma)

        best = 0.0
        stop = 0
        best_state = None
        t0 = time.time()
        print("Start training !!!")

        for epoch in range(10000):
            loss, acc, f1s = train(model, optimizer)
            print(f"epoch={epoch} loss={loss:.4f} acc={acc:.4f} f1={f1s:.4f}")
            scheduler.step()

            if epoch % 10 == 0 and epoch != 0:
                dev_acc, _, _ = test(model, dev_class)
                motif_gate_val = torch.sigmoid(model.motif_gate).item()
                print(f"Dev acc: {dev_acc:.4f} (best: {best:.4f}) | motif gate: {motif_gate_val*100:.2f}%")
                if dev_acc >= best:
                    best = dev_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    stop = 0
                else:
                    stop += 1
                if stop > patience_val:
                    print("Early stop at epoch", epoch, "time", time.time() - t0)
                    break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})


        test_acc, test_std, test_interval = test(model, test_class)
        print("Test acc:", test_acc, "std:", test_std, "interval:", test_interval)
        return test_acc, test_std, test_interval

    # run seeds
    accs, stds, intervals = [], [], []
    for seed_i in range(args.num_seeds):
        a, s, iv = train_eval()
        accs.append(a)
        stds.append(s)
        intervals.append(iv)

    print("Final acc:", float(np.mean(accs)))
    print("Final std:", float(np.mean(stds)))
    print("Final interval:", float(np.mean(intervals)))

    with open(f"{BASE_DIR}/results.txt", "a") as f:
        f.write(json.dumps({
            "dataset": args.dataset,
            "n_way": args.n_way,
            "k_shot": args.k_shot,
            "inductive": args.inductive,
            "num_seeds": args.num_seeds,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.mean(stds)),
            "acc_interval": float(np.mean(intervals)),
        }) + "\n")


if __name__ == "__main__":
    main()
