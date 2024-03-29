import networkx as nx
import matplotlib.pyplot as plt

def plot_graph(G):
    plt.figure()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue')

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

# 创建一个示例图
G = nx.DiGraph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (6, 7), (4, 8)])

# 选择节点1及其所有子节点，创建一个新图
sub_G = get_subgraph(G, 2)
plot_graph(G)
plot_graph(sub_G)
plt.show()


# print("每个节点的 order 标记:")
# print(order)

