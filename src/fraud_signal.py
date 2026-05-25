"""Extract fraud suspicion scores and evaluate detection performance."""

import numpy as np


def compute_suspicion_scores(tree: dict, labels: dict) -> dict:
    """
    For each node in the Gomory-Hu tree, compute a suspicion score defined
    as the minimum edge weight incident to that node in the tree.

    Low weight = weakly connected to the rest of the graph = suspicious.
    A node that can be cut away from the honest cluster by removing a single
    low-weight tree edge is a strong candidate for fraud ring membership.

    Implemented in pure Python — no external libraries.

    Args:
        tree:   Parent dict from gusfield() — {node: parent}.
        labels: Edge weight dict from gusfield() — {node: cut_value_to_parent}.
                (Not to be confused with ground-truth fraud labels.)

    Returns:
        {node: suspicion_score} where a lower score means more suspicious.
        Covers every node that appears in the tree (root + all non-root nodes).
    """
    # Build a mapping from each node to all incident tree edge weights.
    # Each entry in `tree` is one edge: node --weight--> parent.
    # That edge contributes its weight to BOTH endpoints.
    incident: dict = {}

    for node, parent in tree.items():
        weight = labels[node]  # weight of the edge node → parent

        if node not in incident:
            incident[node] = []
        incident[node].append(weight)

        if parent not in incident:
            incident[parent] = []
        incident[parent].append(weight)

    # Suspicion score = minimum incident edge weight for each node.
    # Leaf nodes have exactly one incident edge; internal nodes and the root
    # have multiple — we take the weakest link in every case.
    return {node: min(weights) for node, weights in incident.items()}


def flag_suspicious_nodes(
    scores: dict,
    percentile_threshold: float = 10.0,
) -> set:
    """
    Flag nodes whose suspicion score falls at or below the given percentile.

    Because a low score means weak connectivity (= more suspicious), flagging
    the bottom N% of scores identifies the most isolated nodes in the graph.

    Args:
        scores:               {node: suspicion_score} from compute_suspicion_scores().
        percentile_threshold: Nodes at or below this percentile are flagged.
                              Default 10.0 = bottom 10%.

    Returns:
        Set of flagged node IDs.
    """
    if not scores:
        return set()

    cutoff = float(np.percentile(list(scores.values()), percentile_threshold))
    return {node for node, score in scores.items() if score <= cutoff}


def evaluate(flagged: set, fraud_labels: dict) -> dict:
    """
    Compare flagged nodes against ground truth fraud labels.

    Only nodes that appear in fraud_labels are included in the evaluation —
    unlabelled nodes in `flagged` are silently ignored so that nodes outside
    the labelled dataset do not inflate false-positive counts.

    Args:
        flagged:      Set of node IDs flagged as suspicious.
        fraud_labels: {node: 0_or_1} ground-truth labels.

    Returns:
        Dict with keys:
            precision, recall, f1          — float metrics (rounded to 4dp)
            true_positives                 — flagged and actually fraudulent
            false_positives                — flagged but not fraudulent
            false_negatives                — fraudulent but not flagged
            total_flagged                  — len(flagged ∩ labeled)
            total_fraudulent               — total fraud nodes in labeled set
    """
    labeled = set(fraud_labels.keys())
    actual_fraud = {n for n in labeled if fraud_labels[n] == 1}

    # Restrict to nodes we have ground truth for.
    flagged_labeled = flagged & labeled

    tp = len(flagged_labeled & actual_fraud)
    fp = len(flagged_labeled - actual_fraud)
    fn = len(actual_fraud - flagged_labeled)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_flagged": len(flagged_labeled),
        "total_fraudulent": len(actual_fraud),
    }
