
from CellwithNetworkx import *
from DistanceAnalyzer import *
import matplotlib.pyplot as plt

# Example usage:
swc_file_path = './modelFile/cell1.asc'

num_synapses_to_add = 2000
k = [1000]
cluster_radius = 2.5

bin_list = [0, 2.7, 4.5, 7.4, 12, 20, 33, 55, 90, 148, 245]
# bin_list = [0, 4.5, 12, 33, 90, 245]

for i in range(len(k)):
    cell1 = CellwithNetworkx(swc_file_path)
    # type_list = cell1.add_synapses(num_synapses_to_add)
    type_list = cell1.add_clustered_synapses(num_synapses_to_add, k[i], cluster_radius)
    distance_matrix = cell1.calculate_distance_matrix()
    cell1.visualize_synapses('Before Clustering')
    cell1.visualize_distance()

    analyzer = DistanceAnalyzer(distance_matrix, type_list, bin_list)
    analyzer._calculate_bin_percentage(type_list)
    analyzer.visualize_results()
    plt.show()

# num_epochs = 1000
# analyzer.cluster_shuffle(num_epochs)
# type_list_clustered = analyzer.type_list_clustered
# cell1.set_type_list(np.array(type_list_clustered))

# cell1.visualize_synapses('After Clustering')
# # cell1.visualize_distance()

# def plot_heatmap(type_list_1, type_list_2):
#     # 创建一个2*n的矩阵，用于表示两个type_list的热力图
#     n = len(type_list_1)
#     heatmap_matrix = np.zeros((2, n))

#     # 将'A'表示为1，'B'表示为2
#     for i in range(n):
#         if type_list_1[i] == 'A':
#             heatmap_matrix[0, i] = 1
#         else:
#             heatmap_matrix[0, i] = 2

#         if type_list_2[i] == 'A':
#             heatmap_matrix[1, i] = 1
#         else:
#             heatmap_matrix[1, i] = 2

#     # 绘制热力图
#     plt.figure(figsize=(1000, 2))
#     plt.imshow(heatmap_matrix, cmap='RdYlBu', aspect='auto', interpolation='none')

#     # 设置图例和标签
#     plt.colorbar(ticks=[1, 2], label='Type (A/B)')
#     plt.xlabel('Index')
#     plt.ylabel('List')
#     plt.xticks(np.arange(n))
#     plt.yticks([0, 1], ['type_list_1', 'type_list_2'])

# def plot_binarymap(type_list_1, type_list_2):
#     matrix_1 = np.array([1 if t == 'A' else 0 for t in type_list_1])
#     matrix_2 = np.array([1 if t == 'A' else 0 for t in type_list_2])

#     # 组合两个矩阵成为一个2*n的矩阵
#     combined_matrix = np.vstack((matrix_1, matrix_2))

#     # 设置颜色映射，'A'用黑色表示，'B'用白色表示
#     cmap = plt.get_cmap('binary')

#     # 绘制热力图
#     plt.figure(figsize=(1000, 2))
#     plt.imshow(combined_matrix, cmap=cmap, aspect='auto')

#     # 隐藏坐标轴
#     plt.axis('off')

# # 比较每个位置上两个列表的元素是否一致，并输出不一致的位置个数
# different_positions_count = sum(1 for i, j in enumerate(type_list) if j != type_list_clustered[i])
# print("元素不一致的位置个数为:", different_positions_count)

# # plt.show()

# analyzer.visualize_learning_curve()
# analyzer.visualize_results()
# plt.title('Cluster Shuffle')

# analyzer.cluster_traditional()
# analyzer.visualize_results()
# plt.title('Cluster Traditional')
# plt.show()

plt.show()

