from cell_with_networkx import *
# from distanceAnalyzer import *
import sys 
import json
sys.setrecursionlimit(1000000)

# Example usage:
swc_file_path = './modelFile/cell1.asc'

NUM_SYN_BASAL_EXC = 10042
NUM_SYN_APIC_EXC = 16070

NUM_SYN_BASAL_INH = 1023
NUM_SYN_APIC_INH = 1637
# Is it possible to save the above part so we don't need to run the bg part every time

num_clusters, cluster_radius, distance_to_soma, num_conn_per_preunit = 5, 5, 4, 5
num_syn_per_cluster = 20
bg_syn_freq = 1
pref_ori_dg = 0


import multiprocessing

def your_function(parameter):
    # 创建保存文件夹






def run_processes():
    parameters_list = [param1, param2, param3]  # 以列表形式提供不同的参数

    processes = []
    for param in parameters_list:
        process = multiprocessing.Process(target=your_function, args=(param,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    run_processes()


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


