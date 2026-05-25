# Gomory-Hu Fraud Detection

Graph-based fraud signal extraction using Gusfield's algorithm for Gomory-Hu trees, applied to the IEEE-CIS fraud detection dataset.

## Project structure

```
fraud-graph/
├── data/               # Raw Kaggle CSVs (not committed — add your own)
├── src/
│   ├── max_flow.py     # Edmonds-Karp (from scratch)
│   ├── gomory_hu.py    # Gusfield's algorithm (from scratch)
│   ├── graph_builder.py# IEEE-CIS → adjacency dict
│   ├── fraud_signal.py # Suspicion scores + clustering
│   └── visualize.py    # Pyvis interactive graphs
├── notebooks/
│   └── analysis.ipynb  # Benchmarking + evaluation
├── tests/
│   └── test_algorithms.py
├── requirements.txt
└── README.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

## Data setup

1. Download the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) dataset from Kaggle.
2. Place `train_identity.csv` and `train_transaction.csv` in `data/`.

## Algorithm summary

| Module | Algorithm | Complexity |
|---|---|---|
| `max_flow.py` | Edmonds-Karp (BFS Ford-Fulkerson) | O(VE²) |
| `gomory_hu.py` | Gusfield cut-tree construction | O(n · T_flow) |

**Gusfield (1990):** n-1 max-flow computations build a cut tree where the minimum edge weight on any s-t path equals the global min s-t cut. High suspicion ≈ small min-cut from the honest cluster.
