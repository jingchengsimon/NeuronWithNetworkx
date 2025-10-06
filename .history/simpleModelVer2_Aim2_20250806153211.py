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
import hashlib
import pickle

from utils.graph_utils import create_directed_graph, set_graph_order
from utils.generate_init_firing_utils import generate_init_firing
from utils.distance_utils import distance_synapse_mark_compare, recur_dist_to_soma, recur_dist_to_root
from utils.generate_stim_utils import generate_indices, get_stim_ids, generate_vecstim
from utils.count_spikes import count_spikes
from utils.visualize_utils import visualize_morpho

import sys 
import json
import multiprocessing
from utils.genarate_simu_params_utils import generate_simu_params_REAL
sys.setrecursionlimit(1000000)
sys.path.insert(0, '/G/MIMOlab/Codes/NeuronWithNetworkx/mod')
warnings.simplefilter(action='ignore', category=FutureWarning) # remember update df.append to pd.concat
warnings.simplefilter(action='ignore', category=RuntimeWarning) # RuntimeWarning: invalid value encountered in double_scalars

class CellWithNetworkx:
    def __init__(self, swc_file, bg_exc_freq, bg_inh_freq, SIMU_DURATION, STIM_DURATION, epoch_idx):
        h.load_file("import3d.hoc")

        # h.nrn_load_dll('./mod/nrnmech.dll') # For Windows
        h.nrn_load_dll('./mod/x86_64/.libs/libnrnmech.so') # For Linux/Mac
        h.load_file('./modelFile/L5PCbiophys3withNaCa.hoc')
        h.load_file('./modelFile/L5PCtemplate.hoc')

        self.complex_cell = h.L5PCtemplate(swc_file)
        h.celsius = 37
        
        # h.v_init, h_tstop and h.run (attributes for simulation) are included in gui, so don't forget to import gui
        h.v_init = self.complex_cell.soma[0].e_pas # -90 mV

        self.distance_matrix = None

        self.num_syn_basal_exc = 0
        self.num_syn_apic_exc = 0
        self.num_syn_basal_inh = 0
        self.num_syn_apic_inh = 0
        self.num_syn_soma_inh = 0

        # we should have 2 rnd, the one for positioning should be fixed through the simu
        # while the one for generating spikes should be different for each simu

        # self.rnd = np.random.RandomState(int(time.time()))  
        # random.seed(int(time.time())) 

        self.spk_epoch_idx = epoch_idx

        epoch_idx = 42
        self.rnd = np.random.default_rng(epoch_idx) #np.random.RandomState(42) 
        random.seed(epoch_idx) 
        self.epoch_idx = epoch_idx # random seed for the current epoch

        if bg_exc_freq != 0:
            self.spike_interval = 1000/bg_exc_freq # interval=1000(ms)/f

        spk_rnd = np.random.default_rng(self.spk_epoch_idx)
        ratio = spk_rnd.uniform(0.4, 1.6) 

        self.FREQ_EXC = bg_exc_freq * ratio # Hz, /s
        self.FREQ_INH = self.FREQ_EXC * bg_inh_freq/bg_exc_freq  # Hz, /s
        self.SIMU_DURATION = SIMU_DURATION # 1s 
        self.STIM_DURATION = STIM_DURATION # 1s

        self.syn_param_exc = [0, 0.3, 1.8] # reversal_potential, tau1, tau2, syn_weight (actually we don't use these params, delete later)
        self.syn_param_inh = [-86, 1, 8, 0.00069] #->0.00069 uS = 0.69 nS

        self.sections_soma = [i for i in map(list, list(self.complex_cell.soma))]
        self.sections_basal = [i for i in map(list, list(self.complex_cell.basal))] 
        self.sections_apical = [i for i in map(list, list(self.complex_cell.apical))]
        self.sections_axon = [i for i in map(list, list(self.complex_cell.axon))] # axon is not used in this model
        self.all_sections = self.sections_soma + self.sections_basal + self.sections_apical # + self.sections_axon # ignore axon
        self.all_segments = [seg for sec in h.allsec() for seg in sec] #[seg for sec in self.all_sections for seg in sec] 

        # all_segments_dend = np.array([seg for sec in self.sections_basal + self.sections_apical for seg in sec])
        # all_segments_dend_dict = {i: seg for i, seg in enumerate(all_segments_dend)}
        # all_segments_noaxon = [seg for sec in self.all_sections for seg in sec] 
        # all_segments_noaxon_dict = {i: seg for i, seg in enumerate(all_segments_noaxon)}
        
        # data_list = []
        # for _, seg in all_segments_dend_dict.items():
        #     data_list.append({
        #         'section_name': seg.sec.name(),
        #         'x_position': seg.x
        #     })
        # df = pd.DataFrame(data_list)

        # # # 保存为 CSV
        # df.to_csv('all_segments_dend.csv', index=False)
        
        self.section_synapse_df = pd.DataFrame(columns=['section_id_synapse',
                                                'section_synapse',
                                                'segment_synapse',
                                                'loc',
                                                'type',
                                                'distance_to_soma',
                                                'distance_to_tuft',
                                                'cluster_flag',
                                                'cluster_center_flag',
                                                'cluster_id',
                                                'pre_unit_id',
                                                'region',
                                                'branch_idx',
                                                'syn_w',
                                                'synapse',
                                                'netstim',  
                                                'netcon',
                                                'spike_train',
                                                'spike_train_bg'], dtype=object) # for adding vecstim of different orientation
                                        
        # For clustered synapses
        self.basal_channel_type = None
        self.sec_type = None
        self.distance_to_soma = None
        self.num_clusters = None
        self.num_clusters_sampled = None
        self.cluster_radius = None

        self.input_ratio_basal_apic = None
        self.bg_exc_channel_type = None
        self.initW = None
        self.num_func_group = None
        self.inh_delay = None

        self.num_stim = None
        self.stim_time = None
        self.num_conn_per_preunit = None
        self.num_preunit = None

        self.ori_dg_list = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        self.pref_ori_dg = None

        self.unit_ids = None
        self.indices = None
        self.spt_unit_list = None

        self.num_syn_inh_list = None
        self.num_activated_preunit_list = None

        self.soma_v_array = None
        self.apic_v_array = None
        
        # For tuning curve
        self.num_spikes_df = None 

        self.lock = threading.Lock()

        self.section_df = pd.DataFrame(columns=['parent_id', 
                                                'section_id', 
                                                'parent_name', 
                                                'section_name', 
                                                'length', 
                                                'branch_idx',
                                                'section_type'])
        
        self.root_tuft_idx = self.all_sections.index(self.sections_apical[36])
        self.root_tuft_sec = self.sections_apical[36][0].sec
        # create section_df, directed graph DiG by graph_utils
        self.section_df, self.DiG = create_directed_graph(self.all_sections, self.all_segments, self.section_df)

        # assign the order for each section
        self.class_dict_soma, self.class_dict_tuft = set_graph_order(self.DiG, self.root_tuft_idx)
        self.sec_tuft_idx = list(itertools.chain(*self.class_dict_tuft.values()))

    def add_synapses(self, num_syn_basal_exc, num_syn_apic_exc, num_syn_basal_inh, num_syn_apic_inh, num_syn_soma_inh):
        
        self.num_syn_basal_exc = num_syn_basal_exc
        self.num_syn_apic_exc = num_syn_apic_exc
        self.num_syn_basal_inh = num_syn_basal_inh
        self.num_syn_apic_inh = num_syn_apic_inh
        self.num_syn_soma_inh = num_syn_soma_inh

        # add excitatory synapses
        self.add_single_synapse(num_syn_basal_exc, 'basal', 'exc')
        self.add_single_synapse(num_syn_apic_exc, 'apical', 'exc')
        
        # add inhibitory synapses
        self.add_single_synapse(num_syn_basal_inh, 'basal', 'inh')        
        self.add_single_synapse(num_syn_apic_inh, 'apical', 'inh')
        self.add_single_synapse(num_syn_soma_inh, 'soma', 'inh')
                                   
    def add_inputs(self, folder_path, simu_condition, input_ratio_basal_apic, bg_exc_channel_type, initW, num_func_group, inh_delay, num_stim, num_trials):
        
        self.input_ratio_basal_apic = input_ratio_basal_apic
        self.bg_exc_channel_type = bg_exc_channel_type
        self.initW = initW
        self.num_func_group = num_func_group
        self.inh_delay = inh_delay
        self.num_stim = num_stim

        spt_rnd = np.random.RandomState(self.spk_epoch_idx) 
        self.num_syn_inh_list = [self.num_syn_basal_inh, self.num_syn_apic_inh, self.num_syn_soma_inh]
        
        exc_firing_rate_array, inh_firing_rate_array = generate_init_firing(self.section_synapse_df, self.SIMU_DURATION, self.FREQ_EXC, 
                                                                            self.input_ratio_basal_apic, self.bg_exc_channel_type, self.num_func_group,
                                                                            self.epoch_idx, self.spk_epoch_idx, spat_condition)
        
    
        self.section_synapse_df.to_csv(os.path.join(folder_path, 'section_synapse_df.csv'), index=False)
        
    def add_single_synapse(self, num_syn, region, sim_type):
        
        type = 'A' if sim_type == 'exc' else 'B'
        
        if region == 'basal':
            sections = self.sections_basal
            section_length = np.array(self.section_df.loc[self.section_df['section_type'] == 'dend', 'length'])  
        elif region == 'apical':
            sections = self.sections_apical
            section_length = np.array(self.section_df.loc[self.section_df['section_type'] == 'apic', 'length'])
        elif region == 'soma':
            sections = self.sections_soma
            section_length = np.array(self.section_df.loc[self.section_df['section_type'] == 'soma', 'length'])

        def generate_synapse(_):
            section = random.choices(sections, weights=section_length)[0][0].sec
            section_name = section.psection()['name']
            
            section_id_synapse = self.section_df.loc[self.section_df['section_name'] == section_name, 'section_id'].iat[0]
            # self.section_df[self.section_df['section_name'] == section_name]['section_id'].values[0]
            branch_idx = self.section_df.loc[self.section_df['section_name'] == section_name, 'branch_idx'].iat[0]
            # self.section_df[self.section_df['section_name'] == section_name]['branch_idx'].values[0]   

            loc = self.rnd.uniform()
            segment_synapse = section(loc)
            
            distance_to_soma = recur_dist_to_soma(section, loc)
            distance_to_tuft = recur_dist_to_root(section, loc, self.root_tuft_sec) if section_id_synapse in self.sec_tuft_idx else -1 

            data_to_append = {'section_id_synapse': section_id_synapse,
                            'section_synapse': section,
                            'segment_synapse': segment_synapse,
                            'loc': loc,
                            'type': type,
                            'distance_to_soma': distance_to_soma,
                            'distance_to_tuft': distance_to_tuft,
                            'cluster_flag': -1, 
                            'cluster_center_flag': -1,
                            'cluster_id': -1,
                            'pre_unit_id': -1,
                            'region': region,
                            'branch_idx': branch_idx,
                            'syn_w': None,
                            'synapse': None, 
                            'netstim': None,
                            'netcon': None,
                            'spike_train': [],
                            'spike_train_bg': []}

            with self.lock:
                self.section_synapse_df = pd.concat([self.section_synapse_df, pd.DataFrame([data_to_append], dtype=object)], ignore_index=True) # concat is faster than append
                # self.section_synapse_df = self.section_synapse_df.append(data_to_append, ignore_index=True)

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            list(tqdm(executor.map(generate_synapse, range(num_syn)), total=num_syn))

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
    epoch = params.values()

    # time_tag = time.strftime("%Y%m%d", time.localtime())
    # folder_path = '/G/results/simulation/' + time_tag + '/' + folder_tag

    # folder_path = '/G/results/simulation/' + time_tag + '/' + folder_tag          
    if basal_channel_type == 'AMPANMDA':                                                                                
        simu_folder = sec_type + '_range' + str(distance_to_root) + '_' + spat_condtion + '_' + simu_condition + '_NATURAL_exc1.3_funcgroup2_var2' #_variedW_tau43_addNaK_woAP+Ca_aligned_varyinh' # + '_ratio1' + '_exc1.1-1.3' + '_inh4' + '_failprob0.5' + '_funcgroup10'
    elif basal_channel_type == 'AMPA':
            simu_folder = sec_type + '_range' + str(distance_to_root) + '_' + spat_condtion + '_' + simu_condition + '_NATURAL_exc1.3_AMPA' #variedW_tau43_addNaK_woAP+Ca_aligned_varyinh_AMPA' 
    
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
    
    cell1.add_inputs(folder_path, simu_condition, input_ratio_basal_apic, 
                     bg_exc_channel_type, initW, num_func_group, inh_delay, num_stim, num_trials)

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
    sec_type, spat_cond, dis_to_root, epoch = args
    params_list = generate_simu_params_REAL(sec_type, spat_cond, dis_to_root)
    # for epoch in range(1, 2):
    run_processes(params_list, epoch)

if __name__ == "__main__":

    # # Running for sing-cluster analysis (nonlinearity) 
    # combinations = [
    #     (sec_type, spat_cond, dis_to_root, epoch)
    #     for sec_type in ['basal']
    #     for spat_cond in ['clus']
    #     for dis_to_root in [0] 
    #     for epoch in range(1, 4)
    # ]
    # with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:  # 根据CPU核心数调整 multiprocessing.cpu_count()
    #     executor.map(run_combination, combinations)

    # for batch_idx in range(10, 50): 
        # start_epoch = 1 + batch_idx * 20
        # end_epoch = start_epoch + 20  # 不包含end_epoch

    start_epoch, end_epoch = 1, 2
    combinations = [
        (sec_type, spat_cond, dis_to_root, epoch)
        for sec_type in ['basal']
        for spat_cond in ['clus']
        for dis_to_root in [0]
        for epoch in range(start_epoch, end_epoch)
    ]
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        executor.map(run_combination, combinations)


    # multiprocessing.set_start_method('spawn', force=True) # Use spawn will initiate too many NEURON instances 

    # # Running for multi-cluster analysis
    # for sec_type in ['basal']: # ['basal', 'apical']
    #     for dis_to_root in [0]: # [0, 1, 2]
    #         for spat_cond in ['clus']: # ['clus', 'distr']
    #             params_list = generate_simu_params_REAL(sec_type, spat_cond, dis_to_root)
    #             for epoch in range(1, 11):
    #                 run_processes(params_list, epoch)

