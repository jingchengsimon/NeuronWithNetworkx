from neuron import gui, h
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
import warnings
import random
import numba 
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing
import itertools

from utils.graph_utils import create_directed_graph, set_graph_order
from utils.add_inputs_utils import add_background_exc_inputs, add_background_inh_inputs, add_clustered_inputs
from utils.distance_utils import distance_synapse_mark_compare, recur_dist_to_soma, recur_dist_to_root
from utils.generate_stim_utils import generate_indices, get_stim_ids, generate_vecstim
from utils.count_spikes import count_spikes
from utils.visualize_utils import visualize_morpho

import sys 
import json
import multiprocessing
from utils.genarate_simu_params_utils import generate_simu_params
sys.setrecursionlimit(1000000)
sys.path.insert(0, '/G/MIMOlab/Codes/NeuronWithNetworkx/mod')

warnings.simplefilter(action='ignore', category=FutureWarning) # remember update df.append to pd.concat
warnings.simplefilter(action='ignore', category=RuntimeWarning) # RuntimeWarning: invalid value encountered in double_scalars

from simpleModelVer2 import CellWithNetworkx

# main function
swc_file_path = './modelFile/cell1.asc'

def build_cell(**params):

    NUM_SYN_BASAL_EXC, \
    NUM_SYN_APIC_EXC, \
    NUM_SYN_BASAL_INH, \
    NUM_SYN_APIC_INH, \
    NUM_SYN_SOMA_INH, \
    SIMU_DURATION, \
    STIM_DURATION, \
    simu_condition, \
    spat_condtion, \
    basal_channel_type, \
    sec_type, \
    distance_to_root, \
    num_clusters, \
    cluster_radius, \
    bg_exc_freq, \
    bg_inh_freq, \
    input_ratio_basal_apic, \
    bg_exc_channel_type, \
    initW, \
    num_func_group, \
    inh_delay, \
    num_stim, \
    stim_time, \
    num_conn_per_preunit, \
    num_syn_per_clus, \
    pref_ori_dg, \
    num_trials, \
    folder_tag,\
    epoch= params.values()

    # time_tag = time.strftime("%Y%m%d", time.localtime())
    # folder_path = '/G/results/simulation/' + time_tag + '/' + folder_tag

    simu_folder = sec_type + '_range' + str(distance_to_root) + '_' + spat_condtion + '_' + simu_condition + '_variedW_tau43_addNaK_monoconn_1s_withAP+Ca_aligned' # + '_ratio1' + '_exc1.1-1.3' + '_inh4' + '_failprob0.5' + '_funcgroup10'
    # get the remainder of the folder_tag to 42, use 42 instead of 0 for exact division   
    folder_tag = str(int(folder_tag) % 100) if int(folder_tag) % 100 != 0 else '100'
    folder_path = '/G/results/simulation/' + simu_folder + '/' + folder_tag + '/' + str(epoch)

    simulation_params = {
        'cell model': 'L5PN',
        'NUM_SYN_BASAL_EXC': NUM_SYN_BASAL_EXC,
        'NUM_SYN_APIC_EXC': NUM_SYN_APIC_EXC,
        'NUM_SYN_BASAL_INH': NUM_SYN_BASAL_INH,
        'NUM_SYN_APIC_INH': NUM_SYN_APIC_INH,
        'NUM_SYN_SOMA_INH': NUM_SYN_SOMA_INH,
        'SIMU DURATION': SIMU_DURATION,
        'STIM DURATION': STIM_DURATION,
        'simulation condition': simu_condition,
        'synaptic spatial condition': spat_condtion,
        'basal channel type': basal_channel_type,
        'section type': sec_type,
        'distance from clusters to root': distance_to_root,
        'number of clusters': num_clusters,
        'cluster radius': cluster_radius,
        'background excitatory frequency': bg_exc_freq,
        'background inhibitory frequency': bg_inh_freq,
        'input ratio of basal to apical': input_ratio_basal_apic,
        'background excitatory channel type': bg_exc_channel_type,
        'initial weight of AMPANMDA synapses': initW,
        'number of functional groups': num_func_group,
        'delay of inhibitory inputs': inh_delay,
        'number of stimuli': num_stim,
        'time point of stimulation': stim_time,
        'number of connection per preunit': num_conn_per_preunit,
        'number of synapses per cluster': num_syn_per_clus,
        'number of trials': num_trials,
    }

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    json_filename = os.path.join(folder_path, "simulation_params.json")
    with open(json_filename, 'w') as json_file:
        json.dump(simulation_params, json_file, indent=4)

    cell1 = CellWithNetworkx(swc_file_path, bg_exc_freq, bg_inh_freq, SIMU_DURATION, STIM_DURATION, epoch)
    cell1.add_synapses(NUM_SYN_BASAL_EXC, 
                                NUM_SYN_APIC_EXC, 
                                NUM_SYN_BASAL_INH, 
                                NUM_SYN_APIC_INH,
                                NUM_SYN_SOMA_INH)
    
    cell1.assign_clustered_synapses(basal_channel_type, sec_type, distance_to_root, 
                                    num_clusters, cluster_radius, num_stim, stim_time, 
                                    spat_condtion, num_conn_per_preunit, num_syn_per_clus,
                                    folder_path) 

    cell1.add_inputs(folder_path, simu_condition, input_ratio_basal_apic, 
                     bg_exc_channel_type, initW, num_func_group, inh_delay, num_trials)

def run_processes(parameters_list, epoch):

    processes = []  # Create a new process list for each set of parameters
    for params in parameters_list:
        params_with_epoch = params.copy()
        params_with_epoch['epoch'] = epoch
        process = multiprocessing.Process(target=build_cell, kwargs=params_with_epoch)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()  # Join each batch of processes before moving to the next parameter set

def run_combination(args):
    sec_type, spat_cond, dis_to_root = args
    params_list = generate_simu_params(sec_type, spat_cond, dis_to_root)
    for epoch in range(1, 2):
        run_processes(params_list, epoch)


if __name__ == "__main__":

    # # Running for sing-cluster analysis (nonlinearity) 
    # combinations = [
    #     (sec_type, spat_cond, dis_to_root)
    #     for sec_type in ['basal']
    #     for spat_cond in ['dist']
    #     for dis_to_root in [0] 
    #     # for epoch in range(1, 4)
    # ]
    # with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:  # 根据CPU核心数调整 multiprocessing.cpu_count()
    #     executor.map(run_combination, combinations)



    # multiprocessing.set_start_method('spawn', force=True) # Use spawn will initiate too many NEURON instances 

    # Running for multi-cluster analysis
    for sec_type in ['basal']: # ['basal', 'apical']
        for dis_to_root in [0]: # [0, 1, 2]
            for spat_cond in ['clus']: # ['clus', 'distr']
                params_list = generate_simu_params(sec_type, spat_cond, dis_to_root)
                for epoch in range(1, 2):
                    run_processes(params_list, epoch)

