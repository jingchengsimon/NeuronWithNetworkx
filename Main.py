
from CellwithNetworkx import *
from DistanceAnalyzer import *
import matplotlib.pyplot as plt

import sys 
sys.setrecursionlimit(1000000)

# Example usage:
swc_file_path = './modelFile/cell1.asc'

NUM_SYN_BASAL_EXC = 10042
NUM_SYN_APIC_EXC = 16070

NUM_SYN_BASAL_INH = 1023
NUM_SYN_APIC_INH = 1637

num_syn_clustered, k, cluster_radius = 50, 5, 5

# bin_array = np.array([0, 2.7, 4.5, 7.4, 12, 20, 33, 55, 90, 148, 245])

cell1 = CellwithNetworkx(swc_file_path)
cell1.add_background_synapses(NUM_SYN_BASAL_EXC, 
                              NUM_SYN_APIC_EXC, 
                              NUM_SYN_BASAL_INH, 
                              NUM_SYN_APIC_INH)
cell1.add_clustered_synapses(num_syn_clustered, k, cluster_radius)

# cell1.visualize_synapses
# ('Background + Clustered Synapses')

plt.show()

# type_array = cell1.add_clustered_synapses(num_synapses_to_add,k,cluster_radius)

# start_time = time.time()
# distance_matrix = cell1.calculate_distance_matrix()
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time:.4f} seconds")

# cell1.set_synapse_type()
# type_array = cell1.type_array
# # cell1.visualize_synapses('Before Clustering')

# analyzer = DistanceAnalyzer(distance_matrix, type_array, bin_array)
# analyzer._calculate_bin_percentage(type_array)
# analyzer.visualize_single_result()

# plt.show()

# num_epochs = 10
# analyzer.cluster_shuffle(num_epochs)
# type_array_clustered = analyzer.type_array_clustered
# cell1.set_type_array(type_array_clustered)
# # cell1.visualize_synapses('After Clustering')                     

# analyzer.visualize_learning_curve()
# analyzer.visualize_results()
# plt.show()


