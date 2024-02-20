import numpy as np
import matplotlib.pyplot as plt

def generate_clustered_data(total_distance, point_density, type_list, num_clusters_per_type, min_points, max_points):
    total_points = int(total_distance * point_density)
    point_positions = np.linspace(0, total_distance, total_points)
    type_array = np.array([''] * total_points, dtype=str)

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

            chosen_type = rnd.choice(type_list)
            type_array[nearest_indices] = chosen_type

    # colors = {ptype: np.random.rand(3,) for ptype in type_list}
    # colors = {'A': 'blue', 'B': 'red', 'C': 'green'}
    # color_mapping = [colors[ptype] for ptype in type_array]

    # plt.figure(figsize=(10, 5))
    # plt.scatter(point_positions, [1] * total_points, c=color_mapping, marker='o')
    # plt.xlabel('Position')
    # plt.yticks([])
    # plt.title('Clustered Point Distribution')
    # plt.show()

