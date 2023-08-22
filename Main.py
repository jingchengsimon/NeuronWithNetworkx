
from CellwithNetworkx import *
from DistanceAnalyzer import *
import matplotlib.pyplot as plt

import sys 
sys.setrecursionlimit(1000000)

# Example usage:
swc_file_path = './modelFile/cell1.asc'

NUMSYN_BASAL_EXC = 10042
NUMSYN_APIC_EXC = 16070

NUMSYN_BASAL_INH = 1023
NUMSYN_APIC_INH = 1637

k, cluster_radius = 10000, 5

bin_array = np.array([0, 2.7, 4.5, 7.4, 12, 20, 33, 55, 90, 148, 245])
# bin_array = np.array([0, 4.5, 12, 33, 90, 245])

cell1 = CellwithNetworkx(swc_file_path)
cell1.add_synapses(NUMSYN_BASAL_EXC, NUMSYN_APIC_EXC)

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


