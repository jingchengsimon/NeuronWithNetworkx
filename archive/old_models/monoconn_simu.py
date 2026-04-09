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
import argparse
import multiprocessing
sys.setrecursionlimit(1000000)
sys.path.insert(0, '/G/MIMOlab/Codes/NeuronWithNetworkx/mod')

warnings.simplefilter(action='ignore', category=FutureWarning) # remember update df.append to pd.concat
warnings.simplefilter(action='ignore', category=RuntimeWarning) # RuntimeWarning: invalid value encountered in double_scalars

from simpleModelVer2 import CellWithNetworkx

# main function
swc_file_path = './modelFile/cell1.asc'

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
    NUM_SYN_SOMA_INH = args.num_syn_soma_inh
    SIMU_DURATION = args.simu_duration
    STIM_DURATION = args.stim_duration
    simu_condition = args.simu_condition
    spat_condtion = args.spat_condition
    basal_channel_type = args.basal_channel_type
    sec_type = args.sec_type
    distance_to_root = args.distance_to_root
    num_clusters = args.num_clusters
    cluster_radius = args.cluster_radius
    bg_exc_freq = args.bg_exc_freq
    bg_inh_freq = args.bg_inh_freq
    input_ratio_basal_apic = args.input_ratio_basal_apic
    bg_exc_channel_type = args.bg_exc_channel_type
    initW = args.initW
    num_func_group = args.num_func_group
    inh_delay = args.inh_delay
    num_stim = args.num_stim
    stim_time = args.stim_time
    num_conn_per_preunit = args.num_conn_per_preunit
    num_syn_per_clus = args.num_syn_per_clus
    pref_ori_dg = args.pref_ori_dg
    num_trials = args.num_trials
    folder_tag = args.folder_tag
    epoch = args.epoch
    
    # Random seeds: both default to epoch value
    synapse_pos_seed = args.synapse_pos_seed if args.synapse_pos_seed is not None else epoch
    spike_gen_seed = args.spike_gen_seed if args.spike_gen_seed is not None else epoch

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

    cell1 = CellWithNetworkx(swc_file_path, bg_exc_freq, bg_inh_freq, SIMU_DURATION, STIM_DURATION, 
                            synapse_pos_seed, spike_gen_seed)
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

def run_processes(args_list, epoch):
    """Run multiple processes with different parameter sets"""
    processes = []
    for args in args_list:
        args_copy = argparse.Namespace(**vars(args))
        args_copy.epoch = epoch
        process = multiprocessing.Process(target=build_cell, args=(args_copy,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

def run_combination(combination_args):
    """Run combination of parameters"""
    sec_type, spat_cond, dis_to_root = combination_args
    parser = create_parser()
    
    args_list = []
    args = parser.parse_args([])
    args.sec_type = sec_type
    args.spat_condition = spat_cond
    args.distance_to_root = dis_to_root
    args_list.append(args)
    
    for epoch in range(1, 2):
        run_processes(args_list, epoch)


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
    parser = create_parser()
    for sec_type in ['basal']: # ['basal', 'apical']
        for dis_to_root in [0]: # [0, 1, 2]
            for spat_cond in ['clus']: # ['clus', 'distr']
                args_list = []
                args = parser.parse_args([])
                args.sec_type = sec_type
                args.spat_condition = spat_cond
                args.distance_to_root = dis_to_root
                args_list.append(args)
                for epoch in range(1, 2):
                    run_processes(args_list, epoch)

