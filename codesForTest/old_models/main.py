from cell_with_networkx import *
# from distanceAnalyzer import *
import sys 
import json
import numpy as np
sys.setrecursionlimit(1000000)

# Example usage:
swc_file_path = './modelFile/cell1.asc'

import multiprocessing
import argparse

def create_parser():
    """Create and configure argument parser with default values"""
    parser = argparse.ArgumentParser(description='Neuron simulation parameters')
    
    # Synapse numbers
    parser.add_argument('--num_syn_basal_exc', type=int, default=10042)
    parser.add_argument('--num_syn_apic_exc', type=int, default=16070)
    parser.add_argument('--num_syn_basal_inh', type=int, default=1023)
    parser.add_argument('--num_syn_apic_inh', type=int, default=1637)
    parser.add_argument('--num_syn_soma_inh', type=int, default=150)
    
    # Simulation duration
    parser.add_argument('--simu_duration', type=int, default=1000)
    parser.add_argument('--stim_duration', type=int, default=1000)
    parser.add_argument('--stim_time', type=int, default=500)
    parser.add_argument('--num_stim', type=int, default=1)
    
    # Channel types
    parser.add_argument('--basal_channel_type', type=str, default='AMPANMDA', choices=['AMPANMDA', 'AMPA'])
    parser.add_argument('--bg_exc_channel_type', type=str, default='AMPANMDA', choices=['AMPANMDA', 'AMPA'])
    
    # Cluster parameters
    parser.add_argument('--cluster_radius', type=float, default=5.0)
    parser.add_argument('--num_clusters', type=int, default=1)
    parser.add_argument('--num_syn_per_clus', type=int, default=72)
    parser.add_argument('--num_conn_per_preunit', type=int, default=3)
    
    # Simulation conditions
    parser.add_argument('--simu_condition', type=str, default='invivo', choices=['invivo', 'invitro'])
    parser.add_argument('--spat_condition', type=str, default='clus', choices=['clus', 'distr'])
    parser.add_argument('--sec_type', type=str, default='basal', choices=['basal', 'apical'])
    parser.add_argument('--distance_to_root', type=int, default=0)
    
    # Background input parameters
    parser.add_argument('--bg_exc_freq', type=float, default=1.0)
    parser.add_argument('--bg_inh_freq', type=float, default=4.0)
    parser.add_argument('--input_ratio_basal_apic', type=float, default=1.0)
    
    # Synaptic weight parameters
    parser.add_argument('--initW', type=float, default=0.0004)
    parser.add_argument('--num_func_group', type=int, default=10)
    parser.add_argument('--inh_delay', type=float, default=4.0)
    
    # Other parameters
    parser.add_argument('--pref_ori_dg', type=float, default=0.0)
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--folder_tag', type=str, default='1')
    parser.add_argument('--epoch', type=int, default=1)
    
    # Random seeds
    parser.add_argument('--synapse_pos_seed', type=int, default=None)
    parser.add_argument('--spike_gen_seed', type=int, default=None)
    
    return parser

def build_cell(args):
    """Build and simulate cell with parameters from argparse"""
    
    NUM_SYN_BASAL_EXC = args.num_syn_basal_exc
    NUM_SYN_APIC_EXC = args.num_syn_apic_exc
    NUM_SYN_BASAL_INH = args.num_syn_basal_inh
    NUM_SYN_APIC_INH = args.num_syn_apic_inh
    basal_channel_type = args.basal_channel_type
    sec_type = args.sec_type
    distance_to_root = args.distance_to_root
    num_clusters = args.num_clusters
    cluster_radius = args.cluster_radius
    bg_exc_freq = args.bg_exc_freq
    bg_inh_freq = args.bg_inh_freq
    num_stim = args.num_stim
    num_conn_per_preunit = args.num_conn_per_preunit
    num_preunit = args.num_syn_per_clus * int(np.ceil(args.num_clusters / 3))  # Calculate num_preunit
    pref_ori_dg = args.pref_ori_dg
    num_trials = args.num_trials
    folder_tag = args.folder_tag
    epoch = args.epoch
    
    # Random seeds: both default to epoch value
    synapse_pos_seed = args.synapse_pos_seed if args.synapse_pos_seed is not None else epoch
    spike_gen_seed = args.spike_gen_seed if args.spike_gen_seed is not None else epoch

    # 创建保存文件夹
    time_tag = time.strftime("%Y%m%d_%H", time.localtime())

    # folder_path = './results/simulation/pseudo/' + basal_channel_type + '_' + time_tag + '/' + folder_tag
    folder_path = 'D:/results/simulation/pseudo/' + time_tag + '/' + folder_tag

    simulation_params = {
        'NUM_SYN_BASAL_EXC': NUM_SYN_BASAL_EXC,
        'NUM_SYN_APIC_EXC': NUM_SYN_APIC_EXC,
        'NUM_SYN_BASAL_INH': NUM_SYN_BASAL_INH,
        'NUM_SYN_APIC_INH': NUM_SYN_APIC_INH,
        'cell model': 'L5PN',
        'basal channel type': basal_channel_type,
        'section type': sec_type,
        'distance from basal clusters to root': distance_to_root,
        'number of clusters': num_clusters,
        'cluster radius': cluster_radius,
        'background excitatory frequency': bg_exc_freq,
        'background inhibitory frequency': bg_inh_freq,
        'number of stimuli': num_stim,
        'number of connection per preunit': num_conn_per_preunit,
        'number of preunit': num_preunit,
        'number of trials': num_trials,
    }

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    json_filename = os.path.join(folder_path, "simulation_params.json")
    with open(json_filename, 'w') as json_file:
        json.dump(simulation_params, json_file, indent=4)

    cell1 = CellWithNetworkx(swc_file_path, bg_exc_freq, bg_inh_freq)
    cell1.add_synapses(NUM_SYN_BASAL_EXC, 
                                NUM_SYN_APIC_EXC, 
                                NUM_SYN_BASAL_INH, 
                                NUM_SYN_APIC_INH)
    
    # cell1.assign_clustered_synapses(basal_channel_type, sec_type,
                                    # distance_to_root, num_clusters, 
                                    # cluster_radius, num_stim, 
                                    # num_conn_per_preunit, num_preunit,
                                    # folder_path) 
    
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

def run_processes(args_list):
    """Run multiple processes with different parameter sets"""
    processes = []
    for args in args_list:
        process = multiprocessing.Process(target=build_cell, args=(args,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args_list = [args]
    run_processes(args_list)


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


