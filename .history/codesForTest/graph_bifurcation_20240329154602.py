import networkx as nx
import matplotlib.pyplot as plt

# 创建一个示例图
G = nx.DiGraph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (6, 7), (4, 8)])


# 选择节点1及其所有子节点，创建一个新图
sub_G = get_subgraph(G, 4)

# 打印新图的节点
print("Nodes in subgraph:", sub_G.nodes())


# 统计每个节点的出度
out_degree = dict(G.out_degree())

print("每个节点的出度:")
print(out_degree)

# 计算每个节点到起点的最短路径
shortest_paths = nx.single_source_shortest_path_length(G, source=0)
# print(shortest_paths)

# 将除起点以外出度大于1的节点设为分岔点
fork_points = {node for node, degree in out_degree.items() if node != 0 and degree > 1}

class_dict = {}
for node, order in shortest_paths.items():
    if order not in class_dict:
        class_dict[order] = []
    class_dict[order].append(node)

# # 输出分类结果
#     max_order = max(shortest_paths.values())

# for i in range(max_order + 1):
#         print(f"Class {i}: {class_dict.get(i, [])}")


# 统计每个节点的最短路径中分岔点的个数，并设为 order
class_dict = {}
for node, distance in shortest_paths.items():
    forks_in_path = sum(1 for n in nx.shortest_path(G, source=0)[node][1:-1] if n in fork_points)
    order = forks_in_path
    if order not in class_dict:
        class_dict[order] = []
    class_dict[order].append(node)

max_order = max(class_dict)
for i in range(max_order + 1):
    print(f"Class {i}: {class_dict.get(i, [])}")

# print("每个节点的 order 标记:")
# print(order)

