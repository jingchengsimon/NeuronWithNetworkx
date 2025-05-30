import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
from Bio import Phylo
from Bio.Phylo.PhyloXML import Phylogeny, Clade

# 递归遍历分支并编号
def assign_branch_numbers(clade, branch_list, count=0):
    """
    给树的每个分支编号，并记录在 branch_list 中。
    """
    branch_list.append((count, clade))
    count += 1
    for child in clade.clades:
        count = assign_branch_numbers(child, branch_list, count)
    return count

# 递归移除不需要的分支
def prune_tree(clade):
    clade.clades = [child for child in clade.clades if child in branches_to_keep]
    for child in clade.clades:
        prune_tree(child)

matplotlib.rc('font', size=4)

# 计算 y 坐标
def assign_y_coordinates(clade, y_coords, counter=[0]):
    """
    递归计算每个 clade（branch）的 y 坐标。
    - 叶子节点分配整数坐标
    - 内部节点取子节点 y 坐标的均值
    """
    if clade.is_terminal():  # 叶子节点
        y_coords[clade] = counter[0] + 1 # It actually starts from 1 
        counter[0] += 1
    else:  # 内部节点
        for child in clade.clades:
            assign_y_coordinates(child, y_coords, counter)
        y_coords[clade] = sum(y_coords[child] for child in clade.clades) / len(clade.clades)
        
tree = Phylo.read("L5_morphology.xml", "phyloxml")
# 获取所有分支的编号
branch_list = []  # 保存 (编号, 分支对象)
assign_branch_numbers(tree.root, branch_list)

# 保留 0 到 84 编号的分支
branches_to_keep = {clade for num, clade in branch_list if num in range(85)} #85
prune_tree(tree.root) # Don't pass the branch_list, otherwise it will prune all branches

# 获取所有节点的坐标
depths = tree.depths()  # 获取所有分支的 x 坐标
leaf_nodes = tree.get_terminals()  # 获取叶子节点
y_coords = {leaf: i for i, leaf in enumerate(leaf_nodes)}  # y 坐标索引

# 存储 branch 对应的 y 坐标
y_coords = {}
assign_y_coordinates(tree.root, y_coords)
branch_y_list = [(clade, y) for clade, y in y_coords.items()]
branch_y_list.sort(key=lambda x: int(x[0].name)) # x[0]: clade (Bio.Phylo.PhyloXML.Clade), x[1]: y (int)

root_folder_path = '/G/results/simulation/'
color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
test_folder_path_list = ['basal_range0_clus_invivo_variedW_tau43_addNaK_multiclus_woAP+Ca_aligned_varyinh/2/',
                         'basal_range0_clus_invivo_variedW_tau43_addNaK_multiclus_woAP+Ca_aligned_varyinh/3/',
                         'basal_range0_clus_invivo_variedW_tau43_addNaK_multiclus_woAP+Ca_aligned_varyinh/4/',
                         'basal_range0_clus_invivo_variedW_tau43_addNaK_multiclus_woAP+Ca_aligned_varyinh/5/',
                         'basal_range0_clus_invivo_variedW_tau43_addNaK_multiclus_woAP+Ca_aligned_varyinh/6/',
                         'basal_range0_clus_invivo_variedW_tau43_addNaK_multiclus_woAP+Ca_aligned_varyinh/7/']

for test_folder_path in test_folder_path_list:
    folder_idx_list = [1]
    fig, axes = plt.subplots(1, len(folder_idx_list), figsize=(5*len(folder_idx_list), 5), dpi=150)
    axes = np.atleast_1d(axes) 
    for ax, folder_idx in zip(axes, folder_idx_list):
    
        ax.axvline(x=0, color="red", linestyle="--", linewidth=0.5)
        ax.axvline(x=86, color="red", linestyle="--", linewidth=0.5)
        ax.axvline(x=140, color="red", linestyle="--", linewidth=0.5)
        ax.set_title(f"L5b Basal Dendritic Tree {folder_idx}")

        section_synapse_df = pd.read_csv(root_folder_path + test_folder_path + f'{folder_idx}/section_synapse_df.csv')
        clus_section_df = section_synapse_df[(section_synapse_df['cluster_flag'] == 1)]

        txt_flag_list = [False] * 6
        for _, sec in clus_section_df.iterrows():
            section_id, loc, clus_id = sec['section_id_synapse'], sec['loc'], sec['cluster_id']
            branch = next((clade for num, clade in branch_list if num == section_id), None)
            if branch:
                x = depths.get(branch, 0)  # 获取分支的 x 轴坐标
                y = branch_y_list[section_id][1]  # 获取分支的 y 轴坐标
                branch_len = branch_y_list[section_id][0].branch_length
                # print(f"Branch {section_id}: x = {x}, y = {y}, branch_length = {branch_len}")
                ax.scatter(x-(1-loc)*branch_len, y, color=color_list[clus_id%6], marker="x", s=7)
                if not txt_flag_list[clus_id%6]:
                    ax.text(x-(1-loc)*branch_len, y, f'{clus_id+1}', fontsize=10, color=color_list[clus_id%6])
                    txt_flag_list[clus_id] = True
               
        for clade in tree.find_clades():
            clade.width = 0.1  # Adjust branch width globally
    
        plt.sca(ax) 
        Phylo.draw(tree, axes=ax)

        res_path = os.path.join(root_folder_path, '202505_Morpho')
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        plt.savefig(os.path.join(res_path, f'{folder_idx}_morpho.png'), dpi=300, bbox_inches='tight')