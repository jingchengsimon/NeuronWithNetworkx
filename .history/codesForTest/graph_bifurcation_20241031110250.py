import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

def plot_graph(G):
    plt.figure()
    # pos = nx.spring_layout(G)
    # nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue')

    node_colors = []
    for node in G.nodes:
        if 1 <= node <= 3:
            node_colors.append('skyblue')
        elif 4 <= node:
            node_colors.append('purple')
        else:
            node_colors.append('lightgray')  # Default color for other nodes if needed


    pos = graphviz_layout(G, prog="dot")
    nx.draw(G, pos, node_size=70, node_color='skyblue')
    nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes}, font_size=8, font_color="black")
    
    # 标记节点的 order
    # nx.draw_networkx_labels(G, pos, labels=order, font_size=12, font_color='red')
    
    plt.title("Graph Visualization")
    # plt.show()

def get_subgraph(G, node):
    subgraph_nodes = nx.descendants(G, node) | {node}  # 使用descendants函数获取所有后代节点
    subgraph = G.subgraph(subgraph_nodes)  # 创建一个包含指定节点及其后代节点的子图

    # 打印新图的节点
    print("Nodes in subgraph:", subgraph.nodes())

    return subgraph

def out_degree(G,root_node=0):
    
    # 统计每个节点的出度
    out_degree = dict(G.out_degree())

    print("每个节点的出度:")
    print(out_degree)

    # 计算每个节点到起点的最短路径
    shortest_paths = nx.single_source_shortest_path_length(G, source=root_node)
    # print(shortest_paths)

    # 将除起点以外出度大于1的节点设为分岔点
    fork_points = {node for node, degree in out_degree.items() if node != 0 and degree > 1}

    # class_dict = {}
    # for node, order in shortest_paths.items():
    #     if order not in class_dict:
    #         class_dict[order] = []
    #     class_dict[order].append(node)

    # # 输出分类结果
    #     max_order = max(shortest_paths.values())

    # for i in range(max_order + 1):
    #         print(f"Class {i}: {class_dict.get(i, [])}")


    # 统计每个节点的最短路径中分岔点的个数，并设为 order
    class_dict = {}
    for node, distance in shortest_paths.items():
        forks_in_path = sum(1 for n in nx.shortest_path(G, source=root_node)[node][1:-1] if n in fork_points)
        order = forks_in_path
        if order not in class_dict:
            class_dict[order] = []
        class_dict[order].append(node)

    max_order = max(class_dict)
    for i in range(max_order + 1):
        print(f"Class {i}: {class_dict.get(i, [])}")

# 创建一个示例图
G = nx.DiGraph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (6, 7), (4, 8)])

# 选择节点1及其所有子节点，创建一个新图
sub_G = get_subgraph(G, 2)
out_degree(G, 0)
out_degree(sub_G, 2)
plot_graph(G)
plot_graph(sub_G)
plt.show()


# print("每个节点的 order 标记:")
# print(order)

