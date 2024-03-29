from cell_with_networkx import *
# from distanceAnalyzer import *
import sys 
sys.setrecursionlimit(1000000)

# Example usage:
swc_file_path = './modelFile/cell1.asc'

NUM_SYN_BASAL_EXC = 10042
NUM_SYN_APIC_EXC = 16070

NUM_SYN_BASAL_INH = 1023
NUM_SYN_APIC_INH = 1637
# Is it possible to save the above part so we don't need to run the bg part every time

num_clusters, cluster_radius, distance_to_soma, num_conn_per_preunit = 10, 5, 5, 3
num_syn_per_cluster = 20
bg_syn_freq = 1
pref_ori_dg = 0

# 创建保存文件夹

# folder_path = f'./results/simulation/spt/dist{distance_to_soma}_degree{pref_ori_dg}_nClusters{num_clusters}_exp2syn'
folder_path = f'./results/simulation/pseudo/exp2syn_dist{distance_to_soma}_nClus{num_clusters}_nSyn{num_syn_per_cluster}_r{cluster_radius}_bgFreq{bg_syn_freq}'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
cell1 = CellWithNetworkx(swc_file_path,bg_syn_freq)
cell1.add_synapses(NUM_SYN_BASAL_EXC, 
                            NUM_SYN_APIC_EXC, 
                            NUM_SYN_BASAL_INH, 
                            NUM_SYN_APIC_INH)
cell1.assign_clustered_synapses(num_clusters, cluster_radius, distance_to_soma, num_conn_per_preunit, num_syn_per_cluster) 
# cell1.visualize_synapses(folder_path, 'Background + Clustered Synapses')
cell1.add_inputs(folder_path)


# # tuning curve
# input: 0 45 90.. -> cluster
# cell1.visualize_synapses('Background + Clustered Synapses')

# plt.show()

# type_array = cell1.add_clustered_synapses(num_synapses_to_add,num_clusters,cluster_radius)

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


