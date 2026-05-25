"""Build a weighted transaction graph from the IEEE-CIS fraud detection dataset."""

import pandas as pd
from collections import defaultdict, deque

# Email domains that are too common to be meaningful signals — pairs of cards
# that only share one of these are almost certainly unrelated.
_GENERIC_EMAIL_DOMAINS = frozenset({
    "gmail.com", "yahoo.com", "hotmail.com", "anonymous.com"
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _accumulate_edges(groups, weight: int, edge_weights: defaultdict) -> None:
    """
    For each group of card1 values that share an attribute value, add `weight`
    to every unique pair within that group.

    Args:
        groups:       Iterable of array-likes, each holding the card1 values
                      that share one particular attribute value.
        weight:       Integer contribution to add per co-occurring pair.
        edge_weights: Accumulator dict keyed by (smaller_card, larger_card).
    """
    for cards_in_group in groups:
        # Deduplicate and sort so the canonical key is always (low, high),
        # which avoids counting the same pair twice.
        cards = sorted(set(cards_in_group))
        # O(k^2) per group — fine for small groups, but large groups
        # (e.g. thousands of cards sharing a popular device) can be slow.
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                edge_weights[(cards[i], cards[j])] += weight


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_transaction_graph(
    transaction_path: str,
    identity_path: str,
    min_edge_weight: int = 2,
) -> tuple[dict, dict]:
    """
    Load IEEE-CIS CSVs and construct a weighted card-to-card graph suitable
    for Gusfield's Gomory-Hu tree algorithm.

    Nodes are unique card1 values.  Two cards are connected when they share
    one or more of the following attributes; each attribute type contributes a
    fixed weight to the edge, and contributions accumulate:

        DeviceInfo          +3  (strong hardware fingerprint signal)
        (addr1, addr2)      +2  (billing-address pair)
        P_emaildomain       +1  (purchaser email domain, non-generic only)

    Edges whose total accumulated weight is below `min_edge_weight` are
    discarded to keep the graph sparse and reduce noise.

    Args:
        transaction_path: Path to train_transaction.csv.
        identity_path:    Path to train_identity.csv.
        min_edge_weight:  Drop edges with weight strictly below this value.

    Returns:
        (graph, fraud_labels) where:
        - graph: {card1: {neighbor_card1: weight}} undirected adjacency dict.
                 Both (u→v) and (v→u) are present with the same weight.
        - fraud_labels: {card1: 0_or_1}. A card is labelled 1 if ANY of its
                        transactions carries isFraud == 1.
    """
    # ------------------------------------------------------------------
    # 1. Load and merge
    # ------------------------------------------------------------------
    transactions = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)

    # Left join: every transaction is kept; identity columns are NaN when
    # the transaction has no corresponding identity row.
    df = transactions.merge(identity, on="TransactionID", how="left")

    # Drop rows with a missing card1 — these cannot be assigned to a node.
    df = df.dropna(subset=["card1"])
    # card1 is stored as float64 by pandas (NaN forces float); cast to int
    # so node keys are clean integers throughout.
    df["card1"] = df["card1"].astype(int)

    # ------------------------------------------------------------------
    # 2. Fraud labels
    # ------------------------------------------------------------------
    # A card is fraudulent if it was involved in at least one fraud transaction.
    # groupby.max() gives 1 whenever any row for that card has isFraud == 1.
    fraud_labels: dict = (
        df.groupby("card1")["isFraud"].max().astype(int).to_dict()
    )

    # ------------------------------------------------------------------
    # 3. Accumulate edge weights across attribute types
    # ------------------------------------------------------------------
    edge_weights: defaultdict = defaultdict(int)

    # --- Attribute 1: DeviceInfo (weight +3) ---
    # Group all card1 values that appear together on the same device.
    device_df = df.dropna(subset=["DeviceInfo"])
    device_groups = device_df.groupby("DeviceInfo")["card1"].apply(list)
    _accumulate_edges(device_groups, weight=3, edge_weights=edge_weights)

    # --- Attribute 2: (addr1, addr2) billing-address pair (weight +2) ---
    # Both columns must be present; a partial address is not meaningful.
    addr_df = df.dropna(subset=["addr1", "addr2"]).copy()
    addr_df["addr_combo"] = list(zip(
        addr_df["addr1"].astype(int),
        addr_df["addr2"].astype(int),
    ))
    addr_groups = addr_df.groupby("addr_combo")["card1"].apply(list)
    _accumulate_edges(addr_groups, weight=2, edge_weights=edge_weights)

    # --- Attribute 3: P_emaildomain (weight +1, filtered) ---
    # Step 1: drop missing and generic domains.
    email_df = df.dropna(subset=["P_emaildomain"])
    email_df = email_df[~email_df["P_emaildomain"].isin(_GENERIC_EMAIL_DOMAINS)]

    # Step 2: drop domains shared by fewer than 2 distinct cards — a domain
    # with only one card creates no edges and adds noise.
    domain_card_counts = (
        email_df.groupby("P_emaildomain")["card1"]
        .nunique()
    )
    valid_domains = domain_card_counts[domain_card_counts >= 2].index
    email_df = email_df[email_df["P_emaildomain"].isin(valid_domains)]

    email_groups = email_df.groupby("P_emaildomain")["card1"].apply(list)
    _accumulate_edges(email_groups, weight=1, edge_weights=edge_weights)

    # ------------------------------------------------------------------
    # 4. Apply minimum weight threshold
    # ------------------------------------------------------------------
    # Keep only edges strong enough to be meaningful.  This removes pairs
    # that share only a single weak attribute (e.g. one obscure email domain).
    edge_weights = {
        pair: w for pair, w in edge_weights.items() if w >= min_edge_weight
    }

    # ------------------------------------------------------------------
    # 5. Build adjacency dict (undirected — both directions)
    # ------------------------------------------------------------------
    # Initialise every node (including isolated ones that pass no edges).
    all_cards = sorted(df["card1"].unique())
    graph: dict = {card: {} for card in all_cards}

    for (card_a, card_b), weight in edge_weights.items():
        graph[card_a][card_b] = weight
        graph[card_b][card_a] = weight

    return graph, fraud_labels


def get_largest_connected_component(graph: dict) -> dict:
    """
    Extract the largest connected component from an undirected adjacency dict.

    Uses BFS from scratch — no NetworkX.  Runs in O(V + E).

    Gomory-Hu trees require a connected graph; call this before gusfield() to
    ensure the input is connected.

    Args:
        graph: {node: {neighbor: weight}} undirected adjacency dict.

    Returns:
        Adjacency dict in the same format containing only the nodes and edges
        that belong to the largest connected component.
    """
    unvisited = set(graph.keys())
    best_component: set = set()

    while unvisited:
        # Pick an arbitrary unvisited node as BFS root.
        start = next(iter(unvisited))
        component: set = set()
        queue = deque([start])
        unvisited.discard(start)

        while queue:
            node = queue.popleft()
            component.add(node)
            for neighbor in graph.get(node, {}):
                if neighbor in unvisited:
                    unvisited.discard(neighbor)
                    queue.append(neighbor)

        if len(component) > len(best_component):
            best_component = component

    # Build the subgraph restricted to best_component.
    subgraph: dict = {}
    for node in best_component:
        subgraph[node] = {
            neighbor: weight
            for neighbor, weight in graph[node].items()
            if neighbor in best_component
        }

    return subgraph


# ---------------------------------------------------------------------------
# Quick smoke-test: python -m src.graph_builder <transaction_csv> <identity_csv>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python graph_builder.py <transaction_path> <identity_path>")
        sys.exit(1)

    g, labels = build_transaction_graph(sys.argv[1], sys.argv[2])

    n_nodes = len(g)
    # Each undirected edge is stored twice, so divide by 2.
    n_edges = sum(len(nbrs) for nbrs in g.values()) // 2
    n_fraud = sum(1 for v in labels.values() if v == 1)
    avg_degree = (sum(len(nbrs) for nbrs in g.values()) / n_nodes) if n_nodes else 0.0

    print(f"Nodes (unique card1):     {n_nodes:,}")
    print(f"Edges (after threshold):  {n_edges:,}")
    print(f"Fraudulent nodes:         {n_fraud:,}  ({100 * n_fraud / n_nodes:.1f}%)")
    print(f"Average degree:           {avg_degree:.2f}")

    # Collect all undirected edges (u < v to avoid duplicates).
    all_edges = [
        (u, v, w)
        for u, nbrs in g.items()
        for v, w in nbrs.items()
        if u < v
    ]
    all_edges.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 10 highest-weight edges:")
    for u, v, w in all_edges[:10]:
        u_label = "FRAUD" if labels.get(u) else "ok"
        v_label = "FRAUD" if labels.get(v) else "ok"
        print(f"  card1={u} [{u_label}] — card1={v} [{v_label}]  weight={w}")

    lcc = get_largest_connected_component(g)
    print(f"\nLargest connected component: {len(lcc):,} nodes")
