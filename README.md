# GRAFT — Graph-based fraud ring detection using Gomory-Hu trees

## Background

A Gomory-Hu tree is a weighted spanning tree that compactly encodes all n(n−1)/2 pairwise max-flow (min-cut) values of an undirected graph: for any pair (s, t), the minimum edge weight on the unique s–t path in the tree equals the minimum s–t cut in the original graph, and this is achievable with just n−1 max-flow computations. Recent work continues to push the construction toward nearly-linear time — see the [2025 FOCS paper by Goranci et al.](https://arxiv.org/abs/2507.20354) for the current state of the art. In fraud detection, fraudsters share device fingerprints, billing addresses, and email domains, creating dense subgraphs that can be isolated by cutting only a few edges; a card whose minimum incident tree-edge weight is very low sits at the boundary of such a cluster and is structurally suspicious.

## How it works

- **Graph construction** — cards (`card1`) are nodes; edges are added when two cards share a `DeviceInfo` (+3), a billing-address pair (+2), or a non-generic `P_emaildomain` (+1); contributions accumulate and edges below a minimum weight are dropped.
- **Gusfield's algorithm** — runs n−1 Edmonds-Karp max-flow calls to build the Gomory-Hu tree; no NetworkX flow functions are used.
- **Suspicion scoring** — each node's score is the minimum weight of its incident tree edges; low score = easy to cut away from the honest cluster.
- **Evaluation** — nodes in the bottom N-th percentile of scores are flagged; precision, recall, and F1 are computed against the `isFraud` ground-truth labels.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) on Kaggle.  
Download `train_transaction.csv` and `train_identity.csv` and place them in `data/`.

## Results

| Threshold (%) | Flagged | Precision | Recall | F1 |
|:---:|:---:|:---:|:---:|:---:|
| 5  | — | — | — | — |
| 10 | — | — | — | — |
| 15 | — | — | — | — |
| 20 | — | — | — | — |

*Run `python main.py` after placing the CSVs in `data/` to populate this table.*

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
jupyter notebook notebooks/analysis.ipynb
```
