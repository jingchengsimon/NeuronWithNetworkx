import networkx as nx
import matplotlib.pyplot as plt

# 创建一个示例图
G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5)])

subnode_count = dict(G.degree())

print("每个节点的子节点数量:")
print(subnode_count)

# 计算每个节点到起点的最短路径
shortest_paths = nx.shortest_path_length(G, source=0)

# 找到具有多于一个子节点的节点作为分叉点
fork_points = set()
for node in G.nodes():
    children_count = len(list(G.neighbors(node)))
    if children_count > 1:
        fork_points.add(node)

print(fork_points)

# 根据分岔点的数量重新标记每个点的 order
order = {}
for node, distance in shortest_paths.items():
    forks_in_path = sum(1 for n in nx.shortest_path(G, source=0)[node][1:] if n in fork_points)
    order[node] = distance + forks_in_path

print("每个节点到起点的 order 标记:")
print(order)

# 创建绘图对象
pos = nx.spring_layout(G)  # 为了更好地展示，这里使用了 spring_layout
nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue')

# 标记节点的 order
nx.draw_networkx_labels(G, pos, labels=order, font_size=12, font_color='red')

plt.title("Graph Visualization")
plt.show()