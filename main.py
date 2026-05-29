"""
End-to-end fraud detection pipeline using Gomory-Hu trees.

Usage:
    python main.py --transaction data/train_transaction.csv \\
                   --identity    data/train_identity.csv \\
                   --threshold   10 \\
                   --output      output/fraud_tree.html

    python main.py --synthetic   # run on a generated 30-node graph (no data needed)
"""

import argparse
import os
import random
import time

# ---------------------------------------------------------------------------
# Synthetic graph generator (used when real CSVs are absent)
# ---------------------------------------------------------------------------


def _make_synthetic_graph(n_nodes: int = 30, fraud_rate: float = 0.25, seed: int = 42):
    """
    Generate a synthetic fraud graph for end-to-end pipeline testing.

    Structure: a dense fraud cluster weakly bridged to a legitimate cluster.
    Fraud nodes have high internal edge weights and low-weight bridge edges,
    which should produce low suspicion scores and be flagged correctly.

    Args:
        n_nodes:    Total number of card nodes.
        fraud_rate: Fraction of nodes that are fraudulent.
        seed:       Random seed for reproducibility.

    Returns:
        (graph, fraud_labels) in the same format as build_transaction_graph().
    """
    random.seed(seed)
    nodes = list(range(n_nodes))
    n_fraud = int(n_nodes * fraud_rate)

    fraud_labels = {n: (1 if n < n_fraud else 0) for n in nodes}
    fraud_nodes = [n for n in nodes if fraud_labels[n] == 1]
    legit_nodes = [n for n in nodes if fraud_labels[n] == 0]

    graph: dict = {n: {} for n in nodes}

    # Dense fraud cluster — high internal weights mimic shared device signals.
    for i in range(len(fraud_nodes)):
        for j in range(i + 1, len(fraud_nodes)):
            w = random.randint(5, 10)
            graph[fraud_nodes[i]][fraud_nodes[j]] = w
            graph[fraud_nodes[j]][fraud_nodes[i]] = w

    # Legitimate cluster — moderately connected.
    for i in range(len(legit_nodes)):
        for j in range(i + 1, len(legit_nodes)):
            if random.random() < 0.45:
                w = random.randint(3, 8)
                graph[legit_nodes[i]][legit_nodes[j]] = w
                graph[legit_nodes[j]][legit_nodes[i]] = w

    # Weak bridges: each fraud node touches one random legit node.
    # These low-weight cross-cluster edges create the low min-cut signal.
    for f in fraud_nodes:
        legit_node = random.choice(legit_nodes)
        w = random.randint(1, 2)
        graph[f][legit_node] = w
        graph[legit_node][f] = w

    return graph, fraud_labels


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _print_table(rows: list[dict]) -> None:
    """Print a list of dicts as a fixed-width table."""
    if not rows:
        return
    keys = list(rows[0].keys())
    widths = {k: max(len(str(k)), max(len(str(r[k])) for r in rows)) for k in keys}
    header = "  ".join(str(k).ljust(widths[k]) for k in keys)
    sep = "  ".join("-" * widths[k] for k in keys)
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(str(row[k]).ljust(widths[k]) for k in keys))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gomory-Hu fraud detection pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--transaction",
        default="data/train_transaction.csv",
        help="Path to train_transaction.csv",
    )
    parser.add_argument(
        "--identity",
        default="data/train_identity.csv",
        help="Path to train_identity.csv",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Percentile threshold for flagging suspicious nodes",
    )
    parser.add_argument(
        "--output",
        default="output/fraud_tree.html",
        help="Path to write the interactive HTML visualisation",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Ignore CSV paths and run on a synthetic graph",
    )
    args = parser.parse_args()

    # --- Lazy imports keep startup fast when running --help ---
    from src.graph_builder import (
        build_transaction_graph,
        get_largest_connected_component,
    )
    from src.gomory_hu import gusfield
    from src.fraud_signal import (
        compute_suspicion_scores,
        flag_suspicious_nodes,
        evaluate,
    )
    from src.visualize import visualize_tree

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load or generate graph
    # ------------------------------------------------------------------
    use_synthetic = args.synthetic or not (
        os.path.exists(args.transaction) and os.path.exists(args.identity)
    )

    if use_synthetic:
        print("[info] Running on synthetic data (30 nodes).")
        graph_full, fraud_labels = _make_synthetic_graph()
    else:
        print(f"[info] Loading transaction graph from {args.transaction} ...")
        t0 = time.perf_counter()
        graph_full, fraud_labels = build_transaction_graph(
            args.transaction, args.identity
        )
        print(f"       Loaded in {time.perf_counter() - t0:.2f}s")

    n_nodes = len(graph_full)
    n_edges = sum(len(v) for v in graph_full.values()) // 2
    n_fraud = sum(1 for v in fraud_labels.values() if v == 1)
    print(
        f"\nFull graph  —  nodes: {n_nodes:,}  edges: {n_edges:,}  "
        f"fraud nodes: {n_fraud:,} ({100*n_fraud/max(n_nodes,1):.1f}%)"
    )

    # ------------------------------------------------------------------
    # 2. Largest connected component
    # ------------------------------------------------------------------
    graph = get_largest_connected_component(graph_full)
    lcc_labels = {n: fraud_labels[n] for n in graph if n in fraud_labels}
    lcc_nodes = len(graph)
    lcc_edges = sum(len(v) for v in graph.values()) // 2
    print(f"LCC         —  nodes: {lcc_nodes:,}  edges: {lcc_edges:,}")

    # ------------------------------------------------------------------
    # 3. Gusfield's algorithm
    # ------------------------------------------------------------------
    print("\n[info] Running Gusfield's algorithm ...")
    t0 = time.perf_counter()
    tree, labels = gusfield(graph)
    t_gusfield = time.perf_counter() - t0
    print(f"       Done in {t_gusfield:.4f}s  ({len(tree)} tree edges)")

    # ------------------------------------------------------------------
    # 4. Suspicion scoring
    # ------------------------------------------------------------------
    scores = compute_suspicion_scores(tree, labels)

    # ------------------------------------------------------------------
    # 5. Threshold sweep
    # ------------------------------------------------------------------
    thresholds = [5.0, 10.0, 15.0, 20.0]
    # Always include the user-supplied threshold even if not in the defaults.
    if args.threshold not in thresholds:
        thresholds = sorted(set(thresholds) | {args.threshold})

    rows = []
    for pct in thresholds:
        flagged = flag_suspicious_nodes(scores, percentile_threshold=pct)
        m = evaluate(flagged, lcc_labels)
        rows.append(
            {
                "Threshold (%)": f"{pct:.0f}",
                "Flagged": m["total_flagged"],
                "TP": m["true_positives"],
                "FP": m["false_positives"],
                "FN": m["false_negatives"],
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "F1": f"{m['f1']:.4f}",
            }
        )

    print("\n── Results ──────────────────────────────────────────────────")
    _print_table(rows)

    # ------------------------------------------------------------------
    # 6. Visualisation
    # ------------------------------------------------------------------
    print(f"\n[info] Writing visualisation to {args.output} ...")
    visualize_tree(tree, labels, scores, lcc_labels, output_path=args.output)
    print("       Done.")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    best = max(rows, key=lambda r: float(r["F1"]))
    print(f"\n── Summary ──────────────────────────────────────────────────")
    print(f"  Data source   : {'Synthetic' if use_synthetic else 'IEEE-CIS'}")
    print(f"  LCC size      : {lcc_nodes:,} nodes, {lcc_edges:,} edges")
    print(f"  Gusfield time : {t_gusfield:.4f}s")
    print(f"  Best F1       : {best['F1']}  (at {best['Threshold (%)']}% threshold)")
    print(f"  Visualisation : {args.output}")


if __name__ == "__main__":
    main()
