import networkx as nx

# 创建一个示例图
G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5)])

# 计算每个节点到起点的最短路径
shortest_paths = nx.shortest_path_length(G, source=0)

# 统计分岔点数量并重新标记节点的距离
fork_points = {}
for node, distance in shortest_paths.items():
    forks = 0
    for neighbor in G.neighbors(node):
        if shortest_paths[neighbor] == distance - 1:
            forks += 1
    fork_points[node] = forks

# 根据分岔点数量重新标记节点到起点的距离
for node in fork_points:
    shortest_paths[node] += fork_points[node]

print("节点到起点的距离标记:")
print(shortest_paths)
