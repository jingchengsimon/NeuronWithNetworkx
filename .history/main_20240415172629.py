from cell_with_networkx import *
# from distanceAnalyzer import *
import sys 
import json
sys.setrecursionlimit(1000000)

# Example usage:
swc_file_path = './modelFile/cell1.asc'

import multiprocessing
from utils.genarate_simu_params_utils import generate_simu_params

def build_cell(**params):

    NUM_SYN_BASAL_EXC, \
    NUM_SYN_APIC_EXC, \
    NUM_SYN_BASAL_INH, \
    NUM_SYN_APIC_INH, \
    basal_channel_type, \
    sec_type, \
    distance_to_root, \
    num_clusters, \
    cluster_radius, \
    bg_syn_freq, \
    num_stim, \
    num_conn_per_preunit, \
    pref_ori_dg, \
    num_trials, \
    folder_tag = params.values()

    # 创建保存文件夹
    time_tag = time.strftime("%Y%m%d_%H", time.localtime())

    # folder_path = './results/simulation/pseudo/' + basal_channel_type + '_' + time_tag + '/' + folder_tag
    folder_path = 'D:\Results\simulation\pseudo' + time_tag + '/' + folder_tag

    simulation_params = {
        'cell model': 'L5PN',
        'basal channel type': basal_channel_type,
        'section type': sec_type,
        'distance from basal clusters to root': distance_to_root,
        'number of clusters': num_clusters,
        'cluster radius': cluster_radius,
        'background synapse frequency': bg_syn_freq,
        'number of stimuli': num_stim,
        'number of connection per preunit': num_conn_per_preunit,
        # 'number of preunit': num_preunit,
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
    cell1.assign_clustered_synapses(basal_channel_type, sec_type,
                                    distance_to_root, num_clusters, 
                                    cluster_radius, num_stim, 
                                    num_conn_per_preunit,
                                    folder_path) 
    
    # cell1.visualize_synapses(folder_path, 'Background + Clustered Synapses')
    cell1.add_inputs(folder_path, num_trials)

# def run_threads_or_processes(parameters_list):
#     threads_or_processes = []
#     for params in parameters_list:
#         # 使用 **params 解包字典，并传递给 your_function
#         thread_or_process = threading.Thread(target=your_function, kwargs=params)
#         threads_or_processes.append(thread_or_process)
#         thread_or_process.start()

#     for thread_or_process in threads_or_processes:
#         thread_or_process.join()

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

    params_list = generate_simu_params()
    run_processes(params_list)


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


