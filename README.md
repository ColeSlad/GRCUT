![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

# GRAFT — Graph-based fraud ring detection using Gomory-Hu trees

## Background

A Gomory-Hu tree is a weighted spanning tree that compactly encodes all n(n−1)/2 pairwise max-flow (min-cut) values of an undirected graph: for any pair (s, t), the minimum edge weight on the unique s–t path in the tree equals the minimum s–t cut in the original graph, achievable with just n−1 max-flow computations. Recent work continues to push the construction toward nearly-linear time — see the [2025 FOCS paper by Goranci et al.](https://arxiv.org/abs/2507.20354) for the current state of the art. In fraud detection, fraudsters share device fingerprints, billing addresses, and email domains, creating dense subgraphs that can be isolated by cutting only a few edges; a card whose minimum incident tree-edge weight is very low sits at the boundary of such a cluster and is structurally suspicious.

## How it works

- **Graph construction** — cards (`card1`) are nodes; edges are added when two cards share a `DeviceInfo` (+3), a billing-address pair (+2), or a non-generic `P_emaildomain` (+1); contributions accumulate and edges below a minimum weight are dropped.
- **Gusfield's algorithm** — runs n−1 Edmonds-Karp max-flow calls to build the Gomory-Hu tree; no NetworkX flow functions are used inside the core implementation.
- **Suspicion scoring** — each node's score is the minimum weight of its incident tree edges; low score = easy to cut away from the honest cluster.
- **Evaluation** — nodes in the bottom N-th percentile of scores are flagged; precision, recall, and F1 are computed against the `isFraud` ground-truth labels.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) on Kaggle.  
Download `train_transaction.csv` and `train_identity.csv` and place them in `data/`.  
See [`data/README.md`](data/README.md) for full instructions.

## Results

| Threshold (%) | Flagged | Precision | Recall | F1 |
|:---:|:---:|:---:|:---:|:---:|
| 5  | — | — | — | — |
| 10 | — | — | — | — |
| 15 | — | — | — | — |
| 20 | — | — | — | — |

*Run `python main.py` after placing the CSVs in `data/` to populate this table.*

## Project structure

```
.
├── data/                        # Kaggle CSVs (gitignored — see data/README.md)
│   └── README.md                # Download instructions
├── notebooks/
│   └── analysis.ipynb           # Full benchmark + evaluation pipeline
├── output/                      # Generated HTML visualisations (gitignored)
├── src/
│   ├── __init__.py
│   ├── fraud_signal.py          # Suspicion scoring and precision/recall evaluation
│   ├── gomory_hu.py             # Gusfield's Gomory-Hu tree algorithm (from scratch)
│   ├── graph_builder.py         # IEEE-CIS CSV → adjacency dict + fraud labels
│   ├── max_flow.py              # Edmonds-Karp max-flow algorithm (from scratch)
│   └── visualize.py             # Pyvis interactive HTML visualisations
├── tests/
│   ├── __init__.py
│   └── test_algorithms.py       # 20 unit tests; validates against NetworkX
├── main.py                      # CLI entry point — runs the full pipeline
├── requirements.txt             # Pinned direct dependencies
└── README.md
```

## Algorithm complexity

| Step | Algorithm | Complexity |
|---|---|---|
| Max-flow | Edmonds-Karp (BFS Ford-Fulkerson) | O(VE²) per call |
| Cut tree | Gusfield's algorithm | O(n) × O(VE²) = **O(nVE²)** total |

Gusfield's algorithm requires exactly n−1 max-flow computations with no tree restructuring between steps, making it simpler to implement than the original Gomory-Hu construction while achieving the same asymptotic bound.

The [2025 FOCS paper](https://arxiv.org/abs/2507.20354) improves on this with a nearly-linear O(m polylog n) construction — three orders of magnitude faster on large sparse graphs — but requires substantially more complex data structures. This project prioritises clarity and correctness over raw speed.

## Limitations

- **False positives from shared infrastructure.** Legitimate users can share a device (corporate laptop fleet, university computer lab) or address (apartment building, PO box), causing unrelated cards to receive high edge weights and cluster together. The current edge-weight heuristics have no way to distinguish deliberate sharing from coincidental co-occurrence.
- **Graph sparsity vs. coverage trade-off.** The `min_edge_weight` threshold drops weak connections to keep Gusfield tractable, but this also removes real fraud signals where rings are connected by only a single shared attribute.
- **Min-cut does not capture solo fraud.** A fraudster who acts alone and mimics legitimate spending patterns will have a structurally normal position in the graph; their tree-edge weights will be indistinguishable from honest high-volume cards.
- **Research prototype only.** This is not a production fraud detection system. It lacks streaming updates, latency guarantees, model monitoring, and the operational infrastructure required for deployment.

## Implementation note

`src/max_flow.py` (Edmonds-Karp) and `src/gomory_hu.py` (Gusfield's algorithm) are implemented entirely from scratch using only the Python standard library. NetworkX is used **only** in `tests/test_algorithms.py` for correctness validation and in `src/visualize.py` as a layout helper — it is never called inside the core algorithm files.

## Run it

```bash
# 1. Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run tests
pytest tests/ -v

# 3. Run full pipeline (real data)
python main.py \
  --transaction data/train_transaction.csv \
  --identity    data/train_identity.csv \
  --threshold   10 \
  --output      output/fraud_tree.html

# 4. Run on synthetic data (no CSVs needed)
python main.py --synthetic

# 5. Open the notebook
jupyter lab notebooks/analysis.ipynb
```
