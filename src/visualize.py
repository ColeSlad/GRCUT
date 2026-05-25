"""Interactive visualisation of transaction graphs and Gomory-Hu trees via Pyvis."""

import os


def _score_to_hex(score: float, min_s: float, max_s: float) -> str:
    """
    Map a suspicion score to a hex colour on a red-to-green gradient.

    Low score (most suspicious) → red (#ff0000).
    High score (least suspicious) → green (#00ff00).
    """
    t = (score - min_s) / (max_s - min_s) if max_s != min_s else 0.5
    t = max(0.0, min(1.0, t))
    r = int(255 * (1.0 - t))
    g = int(255 * t)
    return f"#{r:02x}{g:02x}00"


def visualize_graph(graph: dict, output_path: str = "graph.html") -> None:
    """
    Render an interactive Pyvis visualisation of a transaction graph.

    Node size scales with degree; edge width scales with edge weight.
    Outputs a self-contained HTML file — open it in any browser.

    Args:
        graph:       Adjacency dict {node: {neighbor: weight}}.
        output_path: Path to write the HTML file.
    """
    from pyvis.network import Network

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    net = Network(height="800px", width="100%", notebook=False, bgcolor="#1a1a2e",
                  font_color="white")
    net.barnes_hut()

    # Add nodes sized by degree.
    for node, neighbors in graph.items():
        degree = len(neighbors)
        net.add_node(
            node,
            label=str(node),
            title=f"Card: {node}<br>Degree: {degree}",
            size=max(8, 4 * degree),
            color="#4a90d9",
        )

    # Add edges only once per undirected pair (u < v).
    seen = set()
    for u, neighbors in graph.items():
        for v, w in neighbors.items():
            if (v, u) not in seen:
                seen.add((u, v))
                net.add_edge(u, v, value=w, title=f"weight: {w}")

    html = net.generate_html()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def visualize_tree(
    tree: dict,
    labels: dict,
    scores: dict,
    fraud_labels: dict,
    output_path: str = "fraud_tree.html",
) -> None:
    """
    Render an interactive Pyvis visualisation of a Gomory-Hu tree with
    fraud-signal overlays.

    Visual encoding:
    - Node fill: red-to-green gradient by suspicion score (red = most suspicious).
    - Node border: black for nodes confirmed fraudulent in ground truth.
    - Node size: scales with degree in the tree.
    - Edge thickness: scales with min-cut weight (thicker = stronger connection).
    - Hover tooltip: card ID, suspicion score, fraud label, tree degree.

    Args:
        tree:         Parent dict from gusfield() — {node: parent}.
        labels:       Edge weight dict from gusfield() — {node: weight_to_parent}.
        scores:       Suspicion scores from compute_suspicion_scores() — {node: score}.
        fraud_labels: Ground-truth labels — {node: 0_or_1}.
        output_path:  Path to write the self-contained HTML file.
    """
    from pyvis.network import Network

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # All nodes that appear in the tree (root has no parent entry but does
    # appear as a parent value).
    all_nodes = set(tree.keys()) | set(tree.values())

    # Degree in the tree = number of incident tree edges.
    degree: dict = {n: 0 for n in all_nodes}
    for node, parent in tree.items():
        degree[node] += 1
        degree[parent] += 1

    # Colour range for score normalisation.
    min_s = min(scores.values()) if scores else 0.0
    max_s = max(scores.values()) if scores else 1.0

    net = Network(height="800px", width="100%", notebook=False, bgcolor="#1a1a2e",
                  font_color="white")
    net.barnes_hut()

    for node in all_nodes:
        score = scores.get(node, 0.0)
        is_fraud = fraud_labels.get(node, 0) == 1
        d = degree.get(node, 0)
        fill = _score_to_hex(score, min_s, max_s)
        # Black border flags ground-truth fraud so it stands out against the
        # colour gradient (which is our model's prediction, not the label).
        border = "#000000" if is_fraud else fill
        title = (
            f"Card: {node}<br>"
            f"Score: {score:.2f}<br>"
            f"Fraud: {'Yes ✗' if is_fraud else 'No'}<br>"
            f"Tree degree: {d}"
        )
        net.add_node(
            node,
            label=str(node),
            title=title,
            color={
                "background": fill,
                "border": border,
                "highlight": {"background": fill, "border": "#ffffff"},
            },
            size=max(10, 8 * d),
            borderWidth=3 if is_fraud else 1,
        )

    for node, parent in tree.items():
        w = labels[node]
        net.add_edge(
            node,
            parent,
            value=w,
            title=f"min-cut: {w}",
            color={"color": "#888888", "highlight": "#ffffff"},
        )

    html = net.generate_html()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
