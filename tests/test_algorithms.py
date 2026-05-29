"""Validate Edmonds-Karp and Gusfield against hand-crafted graphs and NetworkX."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import networkx as nx

from src.max_flow import edmonds_karp
from src.gomory_hu import gusfield

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _min_cut_from_tree(parent_dict: dict, weight_dict: dict, u, v) -> float:
    """
    Return the minimum edge weight on the unique u-v path in a Gomory-Hu tree.

    This equals the min-cut value between u and v in the original graph.
    Traverses the tree represented as (parent_dict, weight_dict) via BFS.
    """
    # Build undirected adjacency list for the tree.
    adj: dict = {}
    for node, par in parent_dict.items():
        w = weight_dict[node]
        adj.setdefault(node, {})[par] = w
        adj.setdefault(par, {})[node] = w

    # BFS tracking minimum edge weight seen so far on the path from u.
    from collections import deque

    queue = deque([(u, float("inf"))])
    visited = {u}
    while queue:
        curr, min_w = queue.popleft()
        if curr == v:
            return min_w
        for neighbor, w in adj.get(curr, {}).items():
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, min(min_w, w)))
    raise ValueError(f"No path from {u} to {v} in tree — graph may be disconnected.")


def _min_cut_from_nx_tree(T: nx.Graph, u, v) -> float:
    """Return the minimum edge weight on the u-v path in a NetworkX Gomory-Hu tree."""
    path = nx.shortest_path(T, u, v)
    return min(T[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))


def _all_nodes(graph: dict) -> list:
    """Return sorted list of all nodes in an adjacency dict."""
    nodes = set(graph.keys())
    for neighbors in graph.values():
        nodes.update(neighbors.keys())
    return sorted(nodes)


def _build_nx_graph(graph: dict) -> nx.Graph:
    """Convert adjacency dict to an undirected NetworkX graph with 'capacity' attrs."""
    G = nx.Graph()
    for u, neighbors in graph.items():
        for v, cap in neighbors.items():
            # Avoid adding the same undirected edge twice.
            if not G.has_edge(u, v):
                G.add_edge(u, v, capacity=cap)
    return G


# ---------------------------------------------------------------------------
# Edmonds-Karp tests
# ---------------------------------------------------------------------------


class TestEdmondsKarp:
    def test_trivial_two_nodes(self):
        """Single edge: flow equals capacity."""
        graph = {0: {1: 5}, 1: {0: 0}}
        val, flow, _ = edmonds_karp(graph, 0, 1)
        assert val == 5.0
        assert flow[0][1] == 5.0

    def test_triangle_undirected(self):
        """Triangle 0-1-2-0 with unit capacities: max flow between any pair = 2."""
        graph = {
            0: {1: 1, 2: 1},
            1: {0: 1, 2: 1},
            2: {0: 1, 1: 1},
        }
        for s, t in [(0, 1), (0, 2), (1, 2)]:
            val, _, _ = edmonds_karp(graph, s, t)
            assert val == 2.0, f"Expected flow 2 for ({s},{t}), got {val}"

    def test_directed_classic(self):
        """Classic 4-node directed flow network: max flow s→t = 5."""
        # 0→1 cap 3, 0→2 cap 2, 1→3 cap 2, 2→3 cap 3, 1→2 cap 1
        graph = {
            0: {1: 3, 2: 2},
            1: {3: 2, 2: 1},
            2: {3: 3},
            3: {},
        }
        val, flow, _ = edmonds_karp(graph, 0, 3)
        assert val == 5.0

    def test_no_path(self):
        """Source and sink in disconnected components: flow = 0."""
        graph = {0: {1: 10}, 1: {}, 2: {3: 10}, 3: {}}
        val, _, _ = edmonds_karp(graph, 0, 3)
        assert val == 0.0

    def test_path_graph_directed(self):
        """Linear chain 0→1→2→3 each cap 1: bottleneck = 1."""
        graph = {0: {1: 1}, 1: {2: 1}, 2: {3: 1}, 3: {}}
        val, _, _ = edmonds_karp(graph, 0, 3)
        assert val == 1.0

    def test_flow_conservation(self):
        """Check that net in-flow = net out-flow at every internal node."""
        graph = {
            0: {1: 4, 2: 3},
            1: {3: 3, 2: 1},
            2: {3: 4},
            3: {},
        }
        val, flow, _ = edmonds_karp(graph, 0, 3)
        for node in [1, 2]:
            inflow = sum(flow.get(u, {}).get(node, 0) for u in graph)
            outflow = sum(flow.get(node, {}).get(v, 0) for v in graph.get(node, {}))
            assert (
                abs(inflow - outflow) < 1e-9
            ), f"Flow not conserved at node {node}: in={inflow}, out={outflow}"

    def test_residual_reachability(self):
        """After max flow, source side of min cut should not contain sink."""
        graph = {
            0: {1: 1, 2: 1},
            1: {0: 1, 2: 1},
            2: {0: 1, 1: 1},
        }
        _, _, residual = edmonds_karp(graph, 0, 1)
        from collections import deque

        visited = {0}
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v, cap in residual.get(u, {}).items():
                if v not in visited and cap > 0:
                    visited.add(v)
                    queue.append(v)
        assert 1 not in visited, "Sink reachable from source in residual after max flow"

    def test_matches_networkx(self):
        """Edmonds-Karp flow values match NetworkX max_flow for several pairs."""
        graph = {
            0: {1: 3, 2: 2},
            1: {0: 3, 2: 1, 3: 2},
            2: {0: 2, 1: 1, 3: 3},
            3: {1: 2, 2: 3},
        }
        G = _build_nx_graph(graph)
        nodes = _all_nodes(graph)
        for s in nodes:
            for t in nodes:
                if s >= t:
                    continue
                our_val, _, _ = edmonds_karp(graph, s, t)
                nx_val = nx.maximum_flow_value(G, s, t, capacity="capacity")
                assert (
                    abs(our_val - nx_val) < 1e-9
                ), f"Flow mismatch ({s},{t}): ours={our_val}, nx={nx_val}"


# ---------------------------------------------------------------------------
# Gusfield tests
# ---------------------------------------------------------------------------


class TestGusfield:
    def test_trivial_two_nodes(self):
        """Two-node graph: tree has one edge with the single min-cut weight."""
        graph = {0: {1: 7}, 1: {0: 7}}
        parent, weights = gusfield(graph)
        assert len(parent) == 1
        assert len(weights) == 1
        cut_val = next(iter(weights.values()))
        assert cut_val == 7.0

    def test_triangle_all_cuts_equal(self):
        """Triangle with unit caps: Gomory-Hu tree encodes all min-cuts = 2."""
        graph = {
            0: {1: 1, 2: 1},
            1: {0: 1, 2: 1},
            2: {0: 1, 1: 1},
        }
        parent, weights = gusfield(graph)
        nodes = _all_nodes(graph)
        assert len(parent) == 2

        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                tree_cut = _min_cut_from_tree(parent, weights, u, v)
                actual_cut, _, _ = edmonds_karp(graph, u, v)
                assert (
                    abs(tree_cut - actual_cut) < 1e-9
                ), f"Tree min-cut ({u},{v}) = {tree_cut}, actual = {actual_cut}"

    def test_path_graph(self):
        """Path 0-1-2-3, unit caps: all min-cuts = 1."""
        graph = {
            0: {1: 1},
            1: {0: 1, 2: 1},
            2: {1: 1, 3: 1},
            3: {2: 1},
        }
        parent, weights = gusfield(graph)
        nodes = _all_nodes(graph)

        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                tree_cut = _min_cut_from_tree(parent, weights, u, v)
                actual_cut, _, _ = edmonds_karp(graph, u, v)
                assert abs(tree_cut - actual_cut) < 1e-9

    def test_two_cliques_bridge(self):
        """Two triangles connected by a bridge of capacity 1.

        Min-cut within a triangle = 2; across the bridge = 1.
        The Gomory-Hu tree must encode this two-tier structure.
        """
        graph = {
            0: {1: 1, 2: 1},
            1: {0: 1, 2: 1, 3: 1},
            2: {0: 1, 1: 1},
            3: {1: 1, 4: 1, 5: 1},
            4: {3: 1, 5: 1},
            5: {3: 1, 4: 1},
        }
        parent, weights = gusfield(graph)
        nodes = _all_nodes(graph)

        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                tree_cut = _min_cut_from_tree(parent, weights, u, v)
                actual_cut, _, _ = edmonds_karp(graph, u, v)
                assert (
                    abs(tree_cut - actual_cut) < 1e-9
                ), f"({u},{v}): tree={tree_cut}, actual={actual_cut}"

    def test_spanning_tree_structure(self):
        """Output tree must have exactly n-1 edges and include all nodes."""
        graph = {
            0: {1: 3, 2: 2, 3: 1},
            1: {0: 3, 2: 1, 3: 2},
            2: {0: 2, 1: 1, 3: 3},
            3: {0: 1, 1: 2, 2: 3},
        }
        parent, weights = gusfield(graph)
        nodes = _all_nodes(graph)
        assert len(parent) == len(nodes) - 1
        assert len(weights) == len(nodes) - 1
        root = nodes[0]
        assert all(n in parent for n in nodes if n != root)

    def test_bottleneck_six_node(self):
        """Two weight-4 triangles joined by a single weight-1 bridge.

        Graph layout:
            0 - 1 - 2 - 3 - 4
            |       |       |
            +---2---+   5---+   (triangle edges weight 4; bridge 2-3 weight 1)

        Concretely:
            Triangle A: 0-1-2-0, all edges weight 4.
            Triangle B: 3-4-5-3, all edges weight 4.
            Bridge:     2-3, weight 1.

        Expected min-cuts:
            Within triangle A or B: 8  (two edge-disjoint paths of weight 4).
            Across the bridge:       1  (only the single bridge edge can be cut).

        The Gomory-Hu tree must encode both tiers correctly; specifically the
        bridge edge must appear in the tree with weight 1.
        """
        graph = {
            0: {1: 4, 2: 4},
            1: {0: 4, 2: 4},
            2: {0: 4, 1: 4, 3: 1},  # node 2 connects the two triangles
            3: {2: 1, 4: 4, 5: 4},
            4: {3: 4, 5: 4},
            5: {3: 4, 4: 4},
        }
        parent, weights = gusfield(graph)
        nodes = _all_nodes(graph)

        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                tree_cut = _min_cut_from_tree(parent, weights, u, v)
                actual_cut, _, _ = edmonds_karp(graph, u, v)
                assert abs(tree_cut - actual_cut) < 1e-9, (
                    f"({u},{v}): tree={tree_cut}, actual={actual_cut}"
                )

        # Explicitly verify the bottleneck: any cross-cluster pair has cut = 1.
        for u in [0, 1, 2]:
            for v in [3, 4, 5]:
                actual_cut, _, _ = edmonds_karp(graph, u, v)
                assert abs(actual_cut - 1.0) < 1e-9, (
                    f"Cross-bridge min-cut ({u},{v}) should be 1, got {actual_cut}"
                )

    def test_uniform_weights_k4(self):
        """K4 with all edge weights equal to 3.

        For K_n with uniform edge weight w, the min-cut between any pair of
        nodes equals (n-1) * w, because isolating one node requires cutting
        all n-1 edges incident to it.

        For K4 with w=3: expected all-pairs min-cut = 3 * 3 = 9.

        This test verifies that Gusfield produces a valid tree even when the
        flow network has no single bottleneck — every possible spanning tree
        with all edge weights 9 is a correct answer.
        """
        edge_weight = 3
        n = 4
        graph = {i: {j: edge_weight for j in range(n) if j != i} for i in range(n)}
        parent, weights = gusfield(graph)
        nodes = _all_nodes(graph)

        expected_cut = (n - 1) * edge_weight  # = 9

        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                tree_cut = _min_cut_from_tree(parent, weights, u, v)
                assert abs(tree_cut - expected_cut) < 1e-9, (
                    f"K4 uniform: expected cut {expected_cut} for ({u},{v}), "
                    f"got {tree_cut}"
                )


# ---------------------------------------------------------------------------
# Gusfield vs. NetworkX validation
# ---------------------------------------------------------------------------


class TestGisfieldVsNetworkX:
    """Compare our Gusfield tree against NetworkX's gomory_hu_tree.

    We do NOT compare tree structure directly (ties may produce different
    spanning trees) but verify that all-pairs min-cut values are identical.
    """

    def _check_all_pairs(self, graph: dict):
        """Assert all-pairs min-cut matches between our tree and NetworkX."""
        parent, weights = gusfield(graph)
        G = _build_nx_graph(graph)
        T_nx = nx.gomory_hu_tree(G, capacity="capacity")
        nodes = _all_nodes(graph)

        for u in nodes:
            for v in nodes:
                if u >= v:
                    continue
                our_cut = _min_cut_from_tree(parent, weights, u, v)
                nx_cut = _min_cut_from_nx_tree(T_nx, u, v)
                assert (
                    abs(our_cut - nx_cut) < 1e-9
                ), f"All-pairs mismatch ({u},{v}): ours={our_cut}, nx={nx_cut}"

    def test_triangle(self):
        graph = {
            0: {1: 1, 2: 1},
            1: {0: 1, 2: 1},
            2: {0: 1, 1: 1},
        }
        self._check_all_pairs(graph)

    def test_complete_k4(self):
        """K4 with unit capacities."""
        graph = {i: {j: 1 for j in range(4) if j != i} for i in range(4)}
        self._check_all_pairs(graph)

    def test_weighted_4_node(self):
        """4-node graph with varying capacities."""
        graph = {
            0: {1: 3, 2: 2},
            1: {0: 3, 2: 1, 3: 2},
            2: {0: 2, 1: 1, 3: 3},
            3: {1: 2, 2: 3},
        }
        self._check_all_pairs(graph)

    def test_two_cliques_bridge(self):
        graph = {
            0: {1: 1, 2: 1},
            1: {0: 1, 2: 1, 3: 1},
            2: {0: 1, 1: 1},
            3: {1: 1, 4: 1, 5: 1},
            4: {3: 1, 5: 1},
            5: {3: 1, 4: 1},
        }
        self._check_all_pairs(graph)

    def test_5_node_mixed_weights(self):
        """5-node graph with mixed integer capacities."""
        graph = {
            0: {1: 5, 2: 3},
            1: {0: 5, 2: 2, 3: 4},
            2: {0: 3, 1: 2, 4: 6},
            3: {1: 4, 4: 1},
            4: {2: 6, 3: 1},
        }
        self._check_all_pairs(graph)
