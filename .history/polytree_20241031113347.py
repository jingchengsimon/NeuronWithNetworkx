import matplotlib.pyplot as plt
import networkx as nx

# Step 1: Construct a sample graph
G = nx.DiGraph()

# Add nodes and edges based on hypothetical phylogenetic data
edges = [
    (121, "A"), (121, "B"), (121, "C"),
    ("A", "A1"), ("A", "A2"), ("B", "B1"), ("C", "C1"), ("C", "C2")
]
G.add_edges_from(edges)

# Step 2: Set node positions in a tree layout
# NetworkX doesn't directly support hierarchical layout, but we can approximate it
pos = nx.multipartite_layout(G, subset_key=lambda n: 0 if n == 121 else 1)

# Step 3: Draw the phylogenetic tree
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_color="lightgreen", edge_color="grey", node_size=1500, font_size=10, font_weight="bold")
plt.title("Phylogenetic Tree Based on NetworkX Graph")
plt.show()
