# Motif-Enhanced Unsupervised Graph Few-Shot Learning

This repository adds a **3-node motif (triangle) branch** to two unsupervised
graph few-shot node-classification frameworks:

| Folder | Backbone | Description |
|---|---|---|
| [`GRACE_motif/`](GRACE_motif) | GRACE | Graph contrastive pre-training + motif-gated encoder |
| [`STAR_motif/`](STAR_motif) | STAR (KDD'25) | Set-function / optimal-transport few-shot learner + motif encoder with a learnable gate |

The two folders are independent. Run each one from inside its own directory.

Motif pipeline (shared idea): nodes are clustered (KMeans) and closed triangles are
extracted. Each triangle is typed by a WL hash over the cluster labels of its nodes. The
node–motif incidence (TF-IDF weighted) feeds a motif encoder, whose output is fused
into the node embedding through a sigmoid gate. Motif extraction is cached to disk
on the first run for each dataset.

## Data

Planetoid / Coauthor / Amazon / CoraFull / ogbn-arxiv download automatically into
`./dataset/` through `torch_geometric` / `ogb`.

Some datasets need to be placed manually (they are not tracked in git):

- **DBLP** (both folders): put `dblp_network`, `dblp_train.mat`, `dblp_test.mat`,
  `dblp_class_split.json` into `./few_shot_data/`. These are the standard
  few-shot DBLP split files used by GPN / TENT.
- **Cora for STAR_motif**: STAR reads the raw LINQS Cora files. Put `cora.content`
  and `cora.cites` into `STAR_motif/cora/`.
  They are available at https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz.

## GRACE_motif

```bash
cd GRACE_motif
pip install -r requirements.txt   # plus torch_geometric / torch_scatter matching your CUDA
python main.py --dataset Cora        --n_way 2  --k_shot 3
python main.py --dataset CoraFull    --n_way 10 --k_shot 5
python main.py --dataset Coauthor-CS --n_way 5  --k_shot 5
python main.py --dataset ogbn-arxiv  --n_way 5  --k_shot 3
```

Supported datasets: `Cora`, `CiteSeer`, `Amazon-Computer`, `Coauthor-CS`, `CoraFull`, `ogbn-arxiv`, `dblp`.
`--num_seeds` (default 5) controls the number of repeated runs. `CUDA_VISIBLE_DEVICES`
defaults to `0`; export it to use a different GPU.

## STAR_motif

```bash
cd STAR_motif
pip install -r requirements.txt   # plus torch_geometric matching your CUDA
python main.py --dataset Cora        --way 2 --shot 5
python main.py --dataset CoraFull    --way 5 --shot 3
python main.py --dataset Coauthor-CS --way 5 --shot 5 --turns 20
```

Per-dataset hyperparameters are in `fs_config.yaml`. Key arguments are `--gpu`
(default 0), `--turns` (the number of seeds, run as consecutive seeds starting from the
config seed) and `--init_motif_gate` (the initial gate logit, default -2.0). Results are
appended to `./result/<dataset>_res.txt`.

## Acknowledgements

`STAR_motif` is built on the official STAR implementation
([KEAML-JLU/STAR](https://github.com/KEAML-JLU/STAR), MIT license; see
`STAR_motif/LICENSE`):

```
@inproceedings{liu2025enhancing,
  title={Enhancing Unsupervised Graph Few-shot Learning via Set Functions and Optimal Transport},
  author={Liu, Yonghao and Giunchiglia, Fausto and Li, Ximing and Huang, Lan and Feng, Xiaoyue and Guan, Renchu},
  booktitle={KDD},
  year={2025}
}
```

`GRACE_motif` builds on GRACE (Zhu et al., *Deep Graph Contrastive Representation Learning*, 2020).
