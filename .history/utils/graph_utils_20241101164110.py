import networkx as nx
import re
import pandas as pd
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt

from Bio import Phylo
from Bio.Phylo.PhyloXML import Phylogeny, Clade

def create_directed_graph(all_sections, section_df):
    max_string_length = 50

    parent_list, parent_index_list = [], []
    
    for i, section_segment_list in enumerate(all_sections):
        section = section_segment_list[0].sec
        section_id = i
        section_name = section.psection()['name']
        match = re.search(r'\.(.*?)\[', section_name)
        section_type = match.group(1)
        L = section.psection()['morphology']['L']

        parent_list.append(section_name)
        parent_index_list.append(section_id)

        if i == 0:
            parent_name = 'None'
            parent_id = 0
            
        else:
            parent = section.psection()['morphology']['parent'].sec
            parent_name = parent.psection()['name']
            parent_id = parent_index_list[parent_list.index(parent_name)]
        
        # create data
        data_to_append = {'parent_id': parent_id,
                'section_id': section_id,
                'parent_name': parent_name,
                'section_name': section_name,
                'length': L,
                'section_type': section_type}

        # self.section_df = self.section_df.append(data_to_append, ignore_index=True)
        section_df = pd.concat([section_df, pd.DataFrame(data_to_append, index=[0])], ignore_index=True)
        
    # G = nx.from_pandas_edgelist(section_df, source='parent_id', target='section_id',
                                #  create_using=nx.Graph(), edge_attr=True)
    
    DiG = nx.from_pandas_edgelist(section_df, source='parent_id', target='section_id',
                                 create_using=nx.DiGraph(), edge_attr=True)
    
    # sp = dict(nx.all_pairs_shortest_path(G))


    # Convert the NetworkX graph to a PhyloXML tree
    root = 'Root'  # Define the root of your NetworkX graph
    root_clade = nx_to_clade(DiG, root)
    phylo_tree = Phylogeny.from_clade(root_clade, rooted=True)

    # Save the PhyloXML tree to an XML file
    with open("L5_morphology.xml", "w") as xml_file:
        Phylo.write([phylo_tree], xml_file, "phyloxml")

    return section_df, DiG

def nx_to_clade(graph, node):
        """Recursively convert NetworkX nodes to Bio.Phylo Clade nodes."""
        clade = Clade(name=node)
        for child in graph.successors(node):  # Only directed edges from parent to child
            edge_data = graph.get_edge_data(node, child)
            weight = edge_data['weight'] if edge_data else 1.0
            child_clade = nx_to_clade(graph, child)
            child_clade.branch_length = weight
            clade.clades.append(child_clade)
        return clade

def plot_graph(G):
    plt.figure()

    # tree layout
    # Find all descendants of node 121 (apical nexus)
    tuft_nodes = nx.descendants(G, 121) if 121 in G.nodes else set()
    tuft_nodes.add(121)  # Add the apical nexus node itself to the set

    node_colors = []
    for node in G.nodes:
        if node == 0:
            node_colors.append('lightgreen') # lightgreen color for soma node
        elif 1 <= node < 85:
            node_colors.append('skyblue') # skyblue color for basal nodes
        elif node in tuft_nodes:
            node_colors.append('pink')  # pink color for tuft nodes
        else:
            node_colors.append('lightgray')  # lightgray color for nodes outside specified ranges

    # Set positions and draw nodes with specified colors
    pos = graphviz_layout(G, prog="circo")
    nx.draw(G, pos, with_labels=False, node_size=100, node_color=node_colors)

    # Add text labels for each node (representing the node index)
    nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes}, font_size=10, font_color="black")

    
    # pos = nx.spring_layout(G)
    # nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue')

    # 标记节点的 order
    # nx.draw_networkx_labels(G, pos, labels=order, font_size=12, font_color='red')
    
    plt.title("Graph Visualization")
    plt.show()

def set_graph_order(G, root_tuft):
    
    class_dict_soma = calculate_out_degree(G, 0)

    G_tuft = get_subgraph(G, root_tuft)
    class_dict_tuft = calculate_out_degree(G_tuft, root_tuft)

    return class_dict_soma, class_dict_tuft

def get_subgraph(G, node):
    subgraph_nodes = nx.descendants(G, node) | {node}  # 使用descendants函数获取所有后代节点
    subgraph = G.subgraph(subgraph_nodes)  # 创建一个包含指定节点及其后代节点的子图

    return subgraph

def calculate_out_degree(G, root_node=0):
    order_dict = nx.single_source_shortest_path_length(G, root_node)

    out_degree = dict(G.out_degree())
    fork_points = {node for node, degree in out_degree.items() if node != root_node and degree > 1}
    
    ## distance to the soma
    class_dict = {}
    for node in order_dict.keys():
        forks_in_path = sum(1 for n in nx.shortest_path(G, source=root_node)[node][1:-1] if n in fork_points)
        order = forks_in_path
        if order not in class_dict:
            class_dict[order] = []
        class_dict[order].append(node)

    # max_order = max(class_dict)
    # for i in range(max_order + 1):
    #     print(f"Class {i}: {sorted(class_dict.get(i, []))}")

    return class_dict