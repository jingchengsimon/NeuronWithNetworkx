from pathlib import Path
import argparse
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx


def _tree_positions(graph, root):
    children = {
        node: sorted(
            graph.successors(node),
            key=lambda child: (
                graph.nodes[child].get("section_id", child),
                graph.nodes[child].get("segment_idx", 0),
                child,
            ),
        )
        for node in graph.nodes
    }

    x_cursor = 0
    positions = {}

    def place(node, depth):
        nonlocal x_cursor
        node_children = children[node]
        if not node_children:
            positions[node] = (x_cursor, -depth)
            x_cursor += 1
            return positions[node][0]

        child_x = [place(child, depth + 1) for child in node_children]
        positions[node] = ((child_x[0] + child_x[-1]) / 2, -depth)
        return positions[node][0]

    place(root, 0)
    return positions


def _node_colors(graph):
    colors = []
    for node in graph.nodes:
        attrs = graph.nodes[node]
        section_type = attrs.get("section_type")
        section_id = attrs.get("section_id", -1)
        if section_type == "soma":
            colors.append("#78c679")
        elif section_type == "dend":
            colors.append("#74c7e8")
        elif section_type == "apic" and section_id >= 121:
            colors.append("#f3a6bd")
        elif section_type == "apic":
            colors.append("#c9c9c9")
        else:
            colors.append("#dddddd")
    return colors


def main():
    parser = argparse.ArgumentParser(description="Visualize segment-level morphology DiGraph.")
    parser.add_argument("--graph", default="results/morphology/segment_graph/segment_DiG.pkl")
    parser.add_argument("--out", default="results/morphology/segment_graph/Graph_morpho_segment.png")
    parser.add_argument("--with-labels", action="store_true", default=False)
    parser.add_argument("--fig-width", type=float, default=28)
    parser.add_argument("--fig-height", type=float, default=18)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    graph_path = Path(args.graph)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with graph_path.open("rb") as handle:
        graph = pickle.load(handle)

    roots = [node for node, degree in graph.in_degree() if degree == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected one root, found {len(roots)}: {roots}")
    root = roots[0]

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Expected segment graph to be a directed acyclic graph.")

    positions = _tree_positions(graph, root)

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height), dpi=args.dpi)
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        arrows=True,
        arrowsize=3,
        width=0.45,
        edge_color="#222222",
        alpha=0.82,
        min_source_margin=0,
        min_target_margin=0,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_size=8,
        node_color=_node_colors(graph),
        linewidths=0,
    )

    if args.with_labels:
        nx.draw_networkx_labels(
            graph,
            positions,
            ax=ax,
            labels={node: str(node) for node in graph.nodes},
            font_size=2.5,
            font_color="#111111",
        )

    ax.set_axis_off()
    ax.margins(0.025)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"nodes: {graph.number_of_nodes()}")
    print(f"edges: {graph.number_of_edges()}")
    print(f"output: {out_path.resolve()}")


if __name__ == "__main__":
    main()
