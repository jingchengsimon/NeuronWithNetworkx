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

num_clusters = 5
cluster_radius = 5
distance_to_soma = 4
num_conn_per_preunit = 5
num_syn_per_cluster = 20
bg_syn_freq = 1
pref_ori_dg = 0

import multiprocessing

def build_cell(**params):

    NUM_SYN_BASAL_EXC, \
    NUM_SYN_APIC_EXC, \
    NUM_SYN_BASAL_INH, \
    NUM_SYN_APIC_INH, \
    basal_channel_type, \
    distance_to_soma, \
    num_clusters, \
    num_syn_per_cluster, \
    cluster_radius, \
    bg_syn_freq, \
    num_conn_per_preunit, \
    pref_ori_dg, \
    folder_tag = params.values()

    # 创建保存文件夹
    time_tag = time.strftime("%Y%m%d_%H%M", time.localtime())

    # folder_path = f'./results/simulation/spt/dist{distance_to_soma}_degree{pref_ori_dg}_nClusters{num_clusters}_exp2syn'
    # folder_path = f'./results/simulation/pseudo/AMPANMDA_dist{distance_to_soma}_nClus{num_clusters}_nSyn{num_syn_per_cluster}_r{cluster_radius}_bgFreq{bg_syn_freq}_nIn{num_conn_per_preunit}'
    folder_path = './results/simulation/pseudo/' + basal_channel_type + '_' + time_tag + '_' + folder_tag

    simulation_params = {
        'cell model': 'L5PN',
        'basal channel type': basal_channel_type,
        'distance from basal clusters to soma': distance_to_soma,
        'number of clusters': num_clusters,
        'number of synapses in each cluster': num_syn_per_cluster,
        'cluster radius': cluster_radius,
        'background synapse frequency': bg_syn_freq,
        'number of stimuli': num_conn_per_preunit,
    }

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    json_filename = os.path.join(folder_path, "simulation_params.json")
    with open(json_filename, 'w') as json_file:
        json.dump(simulation_params, json_file, indent=4)

    cell1 = CellWithNetworkx(swc_file_path, bg_syn_freq)
    cell1.add_synapses(NUM_SYN_BASAL_EXC, 
                                NUM_SYN_APIC_EXC, 
                                NUM_SYN_BASAL_INH, 
                                NUM_SYN_APIC_INH)
    cell1.assign_clustered_synapses(num_clusters, cluster_radius, distance_to_soma, num_conn_per_preunit, num_syn_per_cluster, basal_channel_type) 
    # cell1.visualize_synapses(folder_path, 'Background + Clustered Synapses')
    cell1.add_inputs(folder_path)

def run_processes(parameters_list):
    # with multiprocessing.Pool() as pool:
    #     pool.map(build_cell, parameters_list)

    processes = []
    for params in parameters_list:
        process = multiprocessing.Process(target=build_cell, kwargs=params)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    # 创建两个不同的参数字典
    param_common = {
        'NUM_SYN_BASAL_EXC': 10042,
        'NUM_SYN_APIC_EXC': 16070,
        'NUM_SYN_BASAL_INH': 1023,
        'NUM_SYN_APIC_INH': 1637,
    }

    param1_diff = {
        'basal channel type': 'AMPANMDA',
        'distance from basal clusters to soma': 3,
        'number of clusters': 10,
        'number of synapses in each cluster': 10,
        'cluster radius': 5,
        'background synapse frequency': 1,
        'number of stimuli': 5,
        'pref_ori_dg': 0,
        'folder_tag': '1'}

    param2_diff = {
        'basal channel type': 'AMPANMDA',
        'distance from basal clusters to soma': 3,
        'number of clusters': 10,
        'number of synapses in each cluster': 20,
        'cluster radius': 5,
        'background synapse frequency': 1,
        'number of stimuli': 1,
        'pref_ori_dg': 0,
        'folder_tag': '2'}
    

    param3_diff = {
        'basal channel type': 'AMPANMDA',
        'distance from basal clusters to soma': 3,
        'number of clusters': 20,
        'number of synapses in each cluster': 10,
        'cluster radius': 5,
        'background synapse frequency': 1,
        'number of stimuli': 1,
        'pref_ori_dg': 0,
        'folder_tag': '3'
    }

    # 构建参数列表
    parameters_list = [
        {**param_common, **param1_diff},
        {**param_common, **param2_diff},
        {**param_common, **param3_diff},
    ]

    run_processes(parameters_list)

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


