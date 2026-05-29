"""Gusfield's algorithm for computing Gomory-Hu trees.

Reference: Gusfield (1990) "Very Simple Methods for All Pairs Network Flow
Analysis", SIAM J. Comput. 19(1):143-155.
"""

from collections import deque
from src.max_flow import edmonds_karp


def _source_side(residual: dict, source) -> set:
    """
    Return all nodes reachable from source via positive-capacity edges.

    After a max-flow computation this is the source side S of the minimum
    cut: every node in S can still be reached from source in the residual
    graph, while the sink cannot.

    Args:
        residual: Residual graph dict from edmonds_karp.
        source:   The source node used in that max-flow call.

    Returns:
        Set of nodes on source's side of the minimum cut.
    """
    visited = {source}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v, cap in residual.get(u, {}).items():
            if v not in visited and cap > 0:
                visited.add(v)
                queue.append(v)
    return visited


def gusfield(graph: dict) -> tuple[dict, dict]:
    """
    Compute a Gomory-Hu tree for an undirected graph using Gusfield's algorithm.

    Gusfield showed that a cut tree with the Gomory-Hu property can be built
    with exactly n-1 max-flow computations and no tree restructuring between
    steps. The resulting tree T satisfies the key property: for any pair
    (s, t), the minimum s-t cut value in the original graph equals the minimum
    edge weight on the unique s-t path in T.

    Assumes the graph is connected. For disconnected graphs, apply per component.

    Args:
        graph: Adjacency dict {node: {neighbor: capacity}}.
               Each undirected edge (u, v) should appear in both graph[u][v]
               and graph[v][u] with the same capacity.

    Returns:
        (parent_dict, weight_dict) where:
        - parent_dict: {node: parent_node} for every non-root node. The root
                       is the smallest node (first in sorted order) and has no
                       entry.
        - weight_dict: {node: weight} mapping each non-root node to the
                       min-cut value (= max-flow) on the edge to its parent.
    """
    nodes = sorted(graph.keys())
    n = len(nodes)

    if n <= 1:
        return {}, {}

    # t[i] = index of the parent of nodes[i] in the Gomory-Hu tree.
    # w[i] = edge weight (min-cut / max-flow value) between nodes[i] and parent.
    # Star initialization: all nodes point to nodes[0] (the root).
    t = [0] * n
    w = [0.0] * n

    for i in range(1, n):
        j = t[i]  # Current parent index of node i.

        # Step 1: Max-flow between nodes[i] and its current parent
        flow_val, _, residual = edmonds_karp(graph, nodes[i], nodes[j])
        w[i] = flow_val

        # Step 2: Source-side of the minimum i-j cut
        # S contains nodes[i] and every node that can still reach nodes[i]'s
        # side after the flow saturates all min-cut edges.
        S = _source_side(residual, nodes[i])

        # Step 3: Redirect later nodes that share parent j but lie in S
        # If node k > i currently points to j and is on i's side of the cut,
        # it is "closer" to i than to j, so redirect it.
        for k in range(i + 1, n):
            if t[k] == j and nodes[k] in S:
                t[k] = i

        # Step 4: Maintain the Gomory-Hu tree invariant
        # If j's own parent (the grandparent of i) lies in S, the arc between
        # i and j must be swapped so the tree correctly encodes all pairwise
        # min cuts. This is the key step that distinguishes Gusfield's
        # construction from a naive star-based cut tree.
        if nodes[t[j]] in S:
            t[i] = t[j]  # i adopts j's former parent.
            t[j] = i  # j's new parent is i.
            w[i] = w[j]  # i takes j's former edge weight.
            w[j] = flow_val  # j takes the freshly computed flow value.

    parent_dict = {nodes[i]: nodes[t[i]] for i in range(1, n)}
    weight_dict = {nodes[i]: w[i] for i in range(1, n)}
    return parent_dict, weight_dict
