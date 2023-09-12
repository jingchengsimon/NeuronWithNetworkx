import numpy as np
import matplotlib.pyplot as plt
from DistanceAnalyzer import *
from sklearn.cluster import KMeans

bin_array = np.array([0, 2.7, 4.5, 7.4, 12, 20, 33, 55, 90, 148, 245])
# 参数配置
total_distance = 1000
point_density = 2
type_list = ['A', 'B', 'C']  # 类型列表
num_clusters_per_type = (5, 11)  # 每种类型的簇数量范围
min_points = 10
max_points = 100
center_type_prob = 1  # 中心点类型概率

def generate_clustered_data(total_distance, point_density, type_list, num_clusters_per_type, min_points, max_points, center_type_prob):
    total_points = int(total_distance * point_density)
    point_positions = np.linspace(0, total_distance, total_points)
    type_array = np.array([''] * total_points, dtype=str)
    distance_matrix = np.abs(np.subtract.outer(point_positions, point_positions))

    num_types = len(type_list)
    cluster_centers = []

    for i in range(num_types):
        num_clusters = np.random.randint(*num_clusters_per_type)
        cluster_centers_type = np.random.choice(point_positions, num_clusters, replace=False)
        cluster_centers.extend([(center, type_list[i]) for center in cluster_centers_type])

    for centers, ptype in cluster_centers:
        idx = np.where(point_positions == centers)[0]
        type_array[idx] = ptype

    rnd = np.random.RandomState(10)

    while '' in type_array:
        for centers, ptype in cluster_centers:
            num_points = np.random.randint(min_points, max_points + 1)
            if np.count_nonzero(type_array == '') < total_points / 2:
                num_points = np.random.randint(min_points, min(max_points, 31))  # 较少空白时，num_points较小
            distances = np.abs(point_positions - centers)
            eligible_indices = np.where(type_array == '')[0]
            nearest_indices = eligible_indices[np.argsort(distances[eligible_indices])[:num_points]]

            non_center_types = [t for t in type_list if t != ptype]
            chosen_type = rnd.choice([ptype] + non_center_types, p=[center_type_prob] + [(1 - center_type_prob) / (num_types - 1)] * (num_types - 1), size=len(nearest_indices))
            type_array[nearest_indices] = chosen_type
            # type_array[nearest_indices] = ptype
            # print('ptype',ptype,',chosen_type',chosen_type)

    # colors = {ptype: np.random.rand(3,) for ptype in type_list}
    colors = {'A': 'blue', 'B': 'red', 'C': 'green'}
    color_mapping = [colors[ptype] for ptype in type_array]

    plt.figure(figsize=(10, 5))
    plt.scatter(point_positions, [1] * total_points, c=color_mapping, marker='o')
    plt.xlabel('Position')
    plt.yticks([])
    plt.title('Clustered Point Distribution')

    return distance_matrix, type_array

distance_matrix, type_array = generate_clustered_data(total_distance, point_density, type_list, num_clusters_per_type, min_points, max_points, center_type_prob)


analyzer = DistanceAnalyzer(distance_matrix, type_array, bin_array)
analyzer._calculate_bin_percentage(type_array)
analyzer.visualize_single_result()

num_epochs = 20
analyzer.cluster_shuffle(num_epochs)
type_array_clustered = analyzer.type_array_clustered

analyzer.visualize_learning_curve()
analyzer.visualize_results()

# analyzer._calculate_bin_percentage(type_array)
# analyzer.visualize_single_result()

plt.show()