import matplotlib.pyplot as plt
import networkx as nx

# Step 1: Construct a sample graph
G = nx.DiGraph()

# Define edges based on hypothetical phylogenetic data
edges = [
    (121, "A"), (121, "B"), (121, "C"),
    ("A", "A1"), ("A", "A2"), ("B", "B1"), ("C", "C1"), ("C", "C2")
]
G.add_edges_from(edges)

# Step 2: Assign a subset level to each node
# Root node (121) is at level 0, its children at level 1, and their children at level 2
for node in G.nodes():
    if node == 121:
        G.nodes[node]['subset'] = 0
    elif node in ["A", "B", "C"]:
        G.nodes[node]['subset'] = 1
    else:
        G.nodes[node]['subset'] = 2

# Step 3: Set node positions in a multipartite layout
pos = nx.multipartite_layout(G, subset_key="subset")

# Step 4: Draw the phylogenetic tree
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_color="lightgreen", edge_color="grey", node_size=100, font_size=10, font_weight="bold")
plt.title("Phylogenetic Tree Based on NetworkX Graph")
plt.show()
