"""Interactive visualisation of transaction graphs and Gomory-Hu trees via Pyvis."""

# NetworkX is allowed here (visualisation helper only, not algorithmic core).
import networkx as nx


def visualize_graph(graph: dict, output_path: str = "graph.html") -> None:
    """
    Render an interactive Pyvis visualisation of a transaction graph.

    Node size scales with degree; edge width scales with capacity/weight.
    Opens as a self-contained HTML file in the browser.

    Args:
        graph:       Adjacency dict {node: {neighbor: capacity}}.
        output_path: Path to write the HTML file.
    """
    # TODO: Implement graph visualisation.
    #
    # Suggested steps:
    # 1. from pyvis.network import Network
    # 2. net = Network(height="750px", width="100%", notebook=False)
    # 3. Add nodes: net.add_node(n, size=degree) for n in graph
    # 4. Add edges: net.add_edge(u, v, value=capacity, title=f"{capacity:.1f}")
    #    (only add once for undirected graphs)
    # 5. net.save_graph(output_path)
    raise NotImplementedError("visualize_graph not yet implemented.")


def visualize_gomory_hu_tree(
    parent_dict: dict,
    weight_dict: dict,
    suspicion: dict | None = None,
    output_path: str = "gomory_hu_tree.html",
) -> None:
    """
    Render an interactive Pyvis visualisation of a Gomory-Hu tree.

    Edge labels show min-cut values.  If suspicion scores are provided, node
    colour interpolates from green (safe) to red (suspicious).

    Args:
        parent_dict:  {node: parent} from gusfield().
        weight_dict:  {node: weight} from gusfield().
        suspicion:    Optional {node: score} from fraud_signal.suspicion_scores().
        output_path:  Path to write the HTML file.
    """
    # TODO: Implement Gomory-Hu tree visualisation.
    #
    # Suggested steps:
    # 1. Build a NetworkX tree from parent_dict / weight_dict.
    # 2. Convert to Pyvis Network.
    # 3. Colour nodes by suspicion score if provided.
    # 4. Label edges with min-cut weights.
    # 5. net.save_graph(output_path)
    raise NotImplementedError("visualize_gomory_hu_tree not yet implemented.")
