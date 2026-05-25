"""Edmonds-Karp algorithm for maximum flow (BFS-based Ford-Fulkerson)."""

from collections import deque


def edmonds_karp(graph: dict, source, sink) -> tuple[float, dict, dict]:
    """
    Compute maximum flow from source to sink using Edmonds-Karp.

    Finds augmenting paths via BFS, guaranteeing O(VE^2) worst-case time.
    Each BFS finds the shortest augmenting path (fewest edges), which bounds
    the total number of augmentations to O(VE).

    Args:
        graph: Adjacency dict {node: {neighbor: capacity}} with non-negative
               capacities. Directed graph; for undirected edges include both
               directions with equal capacity.
        source: The source node (flow originates here).
        sink:   The sink node (flow terminates here).

    Returns:
        (flow_value, flow_dict, residual_graph) where:
        - flow_value:     Total max flow from source to sink (float).
        - flow_dict:      {node: {neighbor: net_flow}} for original edges.
                          Net flow = capacity - remaining residual capacity.
        - residual_graph: Final residual capacities dict. residual[u][v] > 0
                          means u->v still has unused capacity, so v is on the
                          source-side reachable set (used by Gusfield).
    """
    # Collect all nodes including those that only appear as neighbors
    all_nodes = set(graph.keys())
    for u in graph:
        for v in graph[u]:
            all_nodes.add(v)

    # Build residual graph
    # For each original edge (u->v, cap): add forward residual capacity `cap`
    # and ensure a backward edge (v->u) exists for flow cancellation.
    residual: dict = {node: {} for node in all_nodes}
    for u in graph:
        for v, cap in graph[u].items():
            # Accumulate in case the same directed edge appears more than once.
            residual[u][v] = residual[u].get(v, 0) + cap
            # Backward edge starts at 0 if no original edge exists in that direction.
            if u not in residual[v]:
                residual[v][u] = 0

    total_flow = 0.0

    # Augmenting path loop
    while True:
        # BFS: find the shortest (fewest-hop) augmenting path source -> sink.
        parent: dict = {source: None}
        queue = deque([source])

        while queue and sink not in parent:
            u = queue.popleft()
            for v, cap in residual[u].items():
                if v not in parent and cap > 0:
                    parent[v] = u
                    queue.append(v)

        # No augmenting path -> max flow reached.
        if sink not in parent:
            break

        # Find bottleneck: minimum residual capacity along the discovered path.
        path_flow = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual[u][v])
            v = u

        # Push path_flow units along the path, updating the residual graph.
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= path_flow          # Consume forward capacity.
            residual[v][u] = residual[v].get(u, 0) + path_flow  # Open backward capacity.
            v = u

        total_flow += path_flow

    # Reconstruct net flow on original edges 
    # flow[u][v] = original_capacity(u,v) − remaining_residual(u,v)
    # This gives the net flow pushed from u to v on that edge.
    flow: dict = {}
    for u in graph:
        flow[u] = {}
        for v, cap in graph[u].items():
            flow[u][v] = cap - residual[u].get(v, 0)

    return total_flow, flow, residual
