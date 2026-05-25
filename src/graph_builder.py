"""Build a weighted transaction graph from the IEEE-CIS fraud detection dataset."""

import pandas as pd


def build_transaction_graph(
    train_identity_path: str,
    train_transaction_path: str,
) -> dict:
    """
    Load IEEE-CIS CSVs and construct a weighted adjacency dict suitable for
    Edmonds-Karp / Gusfield.

    Each node represents a financial entity (e.g. a card, email domain, or
    device fingerprint).  Each edge weight reflects the total transaction
    volume between two entities, so higher-weight edges indicate stronger
    co-occurrence links that are harder to cut away from the rest of the graph.

    Args:
        train_identity_path:    Path to train_identity.csv.
        train_transaction_path: Path to train_transaction.csv.

    Returns:
        Adjacency dict {node: {neighbor: capacity}} representing an undirected
        weighted graph.  Both (u, v) and (v, u) entries are present with the
        same weight so the dict is ready to pass directly to gusfield().
    """
    # --- Load data ---
    transactions = pd.read_csv(train_transaction_path)
    identity = pd.read_csv(train_identity_path)

    # Merge on TransactionID so every transaction has its identity features.
    df = transactions.merge(identity, on="TransactionID", how="left")

    # TODO: Define nodes.
    # Candidate node types (pick one or combine):
    #   - card1 / card2 (anonymised card identifiers)
    #   - P_emaildomain / R_emaildomain (purchaser / recipient email domains)
    #   - DeviceInfo / DeviceType
    #   - id_30 (OS), id_31 (browser)
    # Example: treat each unique card1 value as a node.

    # TODO: Define edges.
    # Two nodes are connected if they share a transaction.
    # Edge weight = sum of TransactionAmt for all shared transactions.
    # Example skeleton:
    #
    #   graph: dict = {}
    #   for _, row in df.iterrows():
    #       u = row["card1"]
    #       v = row["P_emaildomain"]
    #       w = row["TransactionAmt"]
    #       if pd.isna(u) or pd.isna(v):
    #           continue
    #       graph.setdefault(u, {})
    #       graph.setdefault(v, {})
    #       graph[u][v] = graph[u].get(v, 0.0) + w
    #       graph[v][u] = graph[v].get(u, 0.0) + w
    #   return graph

    # TODO: Add fraud-label enrichment (isFraud column) if needed for signal
    #       extraction downstream.

    raise NotImplementedError(
        "Edge construction logic not yet implemented. "
        "See the TODO comments above for the suggested approach."
    )
