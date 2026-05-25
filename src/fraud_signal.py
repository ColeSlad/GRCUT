"""Extract fraud suspicion scores from a Gomory-Hu tree."""


def suspicion_scores(
    parent_dict: dict,
    weight_dict: dict,
    graph: dict,
) -> dict[object, float]:
    """
    Derive a per-node suspicion score from the Gomory-Hu tree structure.

    Nodes that are weakly connected to the rest of the graph (low min-cut
    values to their nearest cut boundary) are more suspicious: they can be
    isolated from the honest transaction cluster by removing few edges.

    The score is defined as the reciprocal of the minimum edge weight on the
    path from a node to the root of the Gomory-Hu tree.  A small min-cut
    weight → high score → high suspicion.

    Args:
        parent_dict: {node: parent} output from gusfield().
        weight_dict: {node: weight} output from gusfield().
        graph:       Original graph dict (used to identify all nodes).

    Returns:
        {node: score} where score ∈ (0, ∞).  Higher means more suspicious.
    """
    # TODO: Implement score extraction.
    #
    # Suggested steps:
    # 1. Reconstruct the tree as an undirected adjacency list from parent_dict
    #    and weight_dict.
    # 2. For each node, walk the path to the root and track the minimum edge
    #    weight encountered (this is the min cut separating that node from the
    #    root cluster).
    # 3. Return score = 1 / min_weight  (or 1 / (min_weight + ε) to avoid /0).
    # 4. Optionally normalise scores to [0, 1] using min-max scaling.
    raise NotImplementedError("suspicion_scores not yet implemented.")


def cluster_by_cut_threshold(
    parent_dict: dict,
    weight_dict: dict,
    threshold: float,
) -> list[set]:
    """
    Partition nodes into clusters by removing Gomory-Hu tree edges below
    a given min-cut threshold.

    Edges with weight < threshold correspond to weak boundaries in the graph;
    removing them splits the tree into connected components that become the
    fraud clusters.

    Args:
        parent_dict: {node: parent} output from gusfield().
        weight_dict: {node: weight} output from gusfield().
        threshold:   Remove tree edges with weight strictly below this value.

    Returns:
        List of node sets, one per cluster.
    """
    # TODO: Implement threshold-based clustering.
    #
    # Suggested steps:
    # 1. Build an adjacency list from parent_dict / weight_dict, omitting edges
    #    whose weight < threshold.
    # 2. Run BFS/DFS to find connected components in the pruned tree.
    # 3. Return each component as a set of nodes.
    raise NotImplementedError("cluster_by_cut_threshold not yet implemented.")
