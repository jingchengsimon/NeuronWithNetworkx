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

from utils.graph_utils import create_directed_graph, set_graph_order
from utils.add_inputs_utils import add_background_exc_inputs, add_background_inh_inputs, add_clustered_inputs
from utils.distance_utils import distance_synapse_mark_compare
from utils.generate_stim_utils import generate_indices, get_stim_ids, generate_vecstim
from utils.count_spikes import count_spikes

import sys 
sys.path.insert(0, '/home/mimo/Desktop/NeuronWithNetworkx/mod')

# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
# warnings.filterwarnings("ignore", category=numba.NumbaDeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning) # remember update df.append to pd.concat

class CellWithNetworkx:
    def __init__(self, swc_file, bg_exc_freq, bg_inh_freq, DURATION):
        h.load_file("import3d.hoc")
        
        # changing the loc of folder seems to make loading the relative path wrong
        # current_directory = os.path.dirname(__file__)  # 如果在脚本中使用，__file__是指当前脚本的文件名
        # os.chdir(current_directory)
        # relative_path = './mod/nrnmech.dll'
        # nrnmech_path = os.path.abspath(relative_path)
        # h.nrn_load_dll(nrnmech_path)

        # h.load_file('./modelFile/L5PCbiophys3.hoc')
        # h.load_file('./modelFile/L5PCtemplate.hoc')
        h.nrn_load_dll('/home/mimo/Desktop/neuron_reduce/example/mod/nrnmech.dll')
        h.load_file('/home/mimo/Desktop/neuron_reduce/example/modelFile/L5PCbiophys3.hoc')
        h.load_file('/home/mimo/Desktop/neuron_reduce/example/modelFile/L5PCtemplate.hoc')

        self.complex_cell = h.L5PCtemplate(swc_file)
        h.celsius = 37
        
        # h.v_init, h_tstop and h.run (attributes for simulation) are included in gui, so don't forget to import gui
        h.v_init = self.complex_cell.soma[0].e_pas # -90 mV

        self.distance_matrix = None

        self.num_syn_basal_exc = 0
        self.num_syn_apic_exc = 0
        self.num_syn_basal_inh = 0
        self.num_syn_apic_inh = 0
        self.num_syn_clustered = 0

        # we should have 2 rnd, the one for positioning should be fixed through the simu
        # while the one for generating spikes should be different for each simu
        self.rnd = np.random.RandomState() 
        
        if bg_exc_freq != 0:
            self.spike_interval = 1000/bg_exc_freq # interval=1000(ms)/f
        self.FREQ_INH = bg_inh_freq  # Hz, /s
        self.DURATION = DURATION # 60s 

        self.syn_param_exc = [0, 0.3, 1.8, 0.0016] # reversal_potential, tau1, tau2, syn_weight
        self.syn_param_inh = [-86, 1, 8, 0.0008]

        self.sections_basal = [i for i in map(list, list(self.complex_cell.basal))] 
        self.sections_apical = [i for i in map(list, list(self.complex_cell.apical))]
        self.all_sections = [i for i in map(list, list(self.complex_cell.soma))] + self.sections_basal + self.sections_apical   
                                       
        self.section_synapse_df = pd.DataFrame(columns=['section_id_synapse',
                                                'section_synapse',
                                                'segment_synapse',
                                                'synapse', 
                                                'netstim',  
                                                'random',
                                                'netcon',
                                                'loc',
                                                'type',
                                                'cluster_center_flag'
                                                'cluster_id',
                                                'pre_unit_id',
                                                'region']) # for adding vecstim of different orientation
                                        
    
        # self.spike_counts_basal_inh = None
        # self.spike_counts_apic_inh = None

        # For clustered synapses

        # self.num_syn_clustered = None
        self.basal_channel_type = None
        self.sec_type = None
        self.distance_to_soma = None
        self.num_clusters = None
        self.cluster_radius = None

        self.bg_exc_channel_type = None
        self.initW = None
        self.inh_delay = None

        self.num_stim = None
        self.num_syn_per_cluster = None
        self.num_conn_per_preunit = None
        self.num_preunit = None

        self.ori_dg_list = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        self.pref_ori_dg = None

        self.unit_ids = None
        self.indices = None
        self.spt_unit_list = None

        self.soma_v_array = None
        self.dend_v_array = None
        self.dend_i_array = None
        self.dend_nmda_i_array = None
        self.dend_ampa_i_array = None

        self.apic_v_array = None
        self.apic_ica_array = None
        
        # self.soma_ica_array = None

        # For tuning curve
        # self.num_spikes_list = []
        self.num_spikes_df = None 

        self.lock = threading.Lock()

        self.section_df = pd.DataFrame(columns=['parent_id', 
                                                'section_id', 
                                                'parent_name', 
                                                'section_name', 
                                                'length', 
                                                'section_type'])
        
        self.root_tuft = self.all_sections.index(self.sections_apical[36])
        # create section_df, directed graph DiG by graph_utils
        self.section_df, self.DiG = create_directed_graph(self.all_sections, self.section_df)
        # assign the order for each section
        self.class_dict_soma, self.class_dict_tuft = set_graph_order(self.DiG, self.root_tuft)

    def add_synapses(self, num_syn_basal_exc, num_syn_apic_exc, num_syn_basal_inh, num_syn_apic_inh):
        self.num_syn_basal_exc = num_syn_basal_exc
        self.num_syn_apic_exc = num_syn_apic_exc
        self.num_syn_basal_inh = num_syn_basal_inh
        self.num_syn_apic_inh = num_syn_apic_inh
        
        # add excitatory synapses
        self.add_single_synapse(num_syn_basal_exc, 'basal', 'exc')
        self.add_single_synapse(num_syn_apic_exc, 'apical', 'exc')
        
        # add inhibitory synapses
        self.add_single_synapse(num_syn_basal_inh, 'basal', 'inh')        
        self.add_single_synapse(num_syn_apic_inh, 'apical', 'inh')
                           
    def assign_clustered_synapses(self, basal_channel_type, sec_type,
                                  dis_to_root, num_clusters, 
                                  cluster_radius, num_stim, 
                                  num_conn_per_preunit, num_preunit,
                                  folder_path):
        
        num_preunit = 1
        num_conn_per_preunit = 1
        # Number of synapses in each cluster is not fixed
        self.pref_ori_dg, self.unit_ids, indices = generate_indices(self.rnd, num_clusters, 
                                                                    num_conn_per_preunit, num_preunit)
        self.indices = indices

        # Save assignment
        file_name = 'preunit assignment.txt'
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'w') as f:
            for i, index_list in enumerate(indices):
                f.write(f"Cluster_id: {i}, Num_preunits: {len(index_list)}, Preunit_ids: {index_list}\n")
            
        # Number of synapses in each cluster is fixed
        # indices = [[1] * num_syn_per_cluster]
        
        self.basal_channel_type = basal_channel_type
        self.sec_type = sec_type
        self.dis_to_root = dis_to_root
        self.num_clusters = num_clusters
        self.cluster_radius = cluster_radius
        self.num_stim = num_stim
        # self.num_syn_per_cluster = num_syn_per_cluster
        self.num_conn_per_preunit = num_conn_per_preunit
        self.num_preunit = num_preunit

        # sec_syn_bg_exc_df = self.section_synapse_df[self.section_synapse_df['type'] == 'A']

        if sec_type == 'basal':
            dis_list = self.class_dict_soma.get(dis_to_root, [])
            sections_k_distance = [section[0].sec for i, section in enumerate(self.all_sections) if i in dis_list]
        elif sec_type == 'apical':
            dis_list = self.class_dict_tuft.get(dis_to_root, []) 
            sections_k_distance = [section[0].sec for i, section in enumerate(self.all_sections) if i in dis_list]
        elif sec_type == 'basal_apical':
            dis_list_soma = self.class_dict_soma.get(dis_to_root, [])
            dis_list_tuft = self.class_dict_tuft.get(dis_to_root, [])

            sections_k_dis_soma = [section[0].sec for i, section in enumerate(self.all_sections) if i in dis_list_soma]
            sections_k_dis_tuft = [section[0].sec for i, section in enumerate(self.all_sections) if i in dis_list_tuft]
       
        # sections_k_distance = [section[0].sec for i, section in enumerate(self.all_sections) if i in dis_list]
        
        for i in range(num_clusters):
            sec_syn_bg_exc_df = self.section_synapse_df[(self.section_synapse_df['type'] == 'A')]

            # for basal_apical case
            if sec_type == 'basal_apical':
                if i <= num_clusters//2:
                    sections_k_distance = sections_k_dis_soma 
                    sec_syn_bg_exc_ordered_df = self.section_synapse_df[
                        (self.section_synapse_df['section_synapse'].isin(sections_k_distance)) & 
                        (self.section_synapse_df['type'] == 'A') &
                        (self.section_synapse_df['region'] == 'basal')]
                else:
                    sections_k_distance = sections_k_dis_tuft
                    sec_syn_bg_exc_ordered_df = self.section_synapse_df[
                        (self.section_synapse_df['section_synapse'].isin(sections_k_distance)) & 
                        (self.section_synapse_df['type'] == 'A') &
                        (self.section_synapse_df['region'] == 'apical')]

            # for common case
            else:
                sec_syn_bg_exc_ordered_df = self.section_synapse_df[
                    (self.section_synapse_df['section_synapse'].isin(sections_k_distance)) & 
                    (self.section_synapse_df['type'] == 'A') &
                    (self.section_synapse_df['region'] == sec_type)]

            index_list = indices[i]
            # index_list = indices[0]
            num_syn_per_cluster = len(index_list)
            
            # use the rnd for positioning
            syn_ctr = sec_syn_bg_exc_ordered_df.loc[self.rnd.choice(sec_syn_bg_exc_ordered_df.index)]
            
            # the distance of the center synapse of the cluster to the soma
            cluster_distance = h.distance(syn_ctr['segment_synapse'])

            # assign the center as clustered synapse
            self.section_synapse_df.loc[syn_ctr.name, 'type'] = 'C'
            self.section_synapse_df.loc[syn_ctr.name, 'is_cluster_center'] = 1
            self.section_synapse_df.loc[syn_ctr.name, 'cluster_distance'] = cluster_distance
            self.section_synapse_df.loc[syn_ctr.name, 'cluster_id'] = i
            try:
                self.section_synapse_df.loc[syn_ctr.name, 'pre_unit_id'] = index_list[0]
            except IndexError:
                self.section_synapse_df.loc[syn_ctr.name, 'pre_unit_id'] = -1
       
    def add_inputs(self, folder_path, bg_exc_channel_type, initW, inh_delay, num_trials):

        self.bg_exc_channel_type = bg_exc_channel_type
        self.initW = initW
        self.inh_delay = inh_delay

        self.section_synapse_df.to_csv(os.path.join(folder_path, 'section_synapse_df.csv'), index=False)

        ori_dg_list, unit_ids, num_stims = self.ori_dg_list, self.unit_ids, 2
        # 创建一个空的 DataFrame
        self.num_spikes_df = pd.DataFrame(index=range(1, num_stims + 1), columns=ori_dg_list)
        spt_unit_list = generate_vecstim(unit_ids, self.num_stim, folder_path)
        
        num_syn_inh_list = [self.num_syn_basal_inh, self.num_syn_apic_inh]
        
        # create an ndarray to store the voltage of each cluster of each trial 
        num_time_points = 1 + 40 * self.DURATION
        
        num_activated_preunit_list = range(0, 100+5, 5) # currently don't exceed 100
        num_aff_fibers = len(num_activated_preunit_list)

        self.soma_v_array = np.zeros((num_time_points, num_aff_fibers, num_trials))
        self.apic_v_array = np.zeros((num_time_points, num_aff_fibers, num_trials))
        self.apic_ica_array = np.zeros((num_time_points, num_aff_fibers, num_trials))

        self.dend_v_array = np.zeros((self.num_clusters, num_time_points, num_aff_fibers, num_trials))
        self.dend_i_array = np.zeros((self.num_clusters, num_time_points, num_aff_fibers, num_trials))
        self.dend_nmda_i_array = np.zeros((self.num_clusters, num_time_points, num_aff_fibers, num_trials))
        self.dend_ampa_i_array = np.zeros((self.num_clusters, num_time_points, num_aff_fibers, num_trials))
        
        for num_activated_preunit in num_activated_preunit_list[:1]:
            
            spt_unit_list_truncated = spt_unit_list
            
            add_clustered_inputs(self.section_synapse_df, 
                                 self.syn_param_exc, 
                                 self.num_clusters, 
                                 self.basal_channel_type,   
                                 self.initW,
                                 spt_unit_list_truncated, 
                                 self.lock) 
            
            for num_trial in range(num_trials):

                add_background_exc_inputs(self.section_synapse_df, 
                                        self.syn_param_exc, 
                                        self.spike_interval, 
                                        self.bg_exc_channel_type,
                                        self.initW,
                                        self.lock)
                
                add_background_inh_inputs(self.section_synapse_df, 
                                        self.syn_param_inh, 
                                        self.DURATION, 
                                        self.FREQ_INH, 
                                        num_syn_inh_list, 
                                        self.inh_delay,
                                        self.lock)
                
                ori_dg = stim_id = stim_index = 1
                # stim_index = np.where(stim_ids == stim_id)[0][0] + 1
            
                # Run the simulation
                num_aff_fiber = num_activated_preunit_list.index(num_activated_preunit)
                self.run_simulation(num_aff_fiber, num_trial)

        np.save(os.path.join(folder_path,'soma_v_array.npy'), self.soma_v_array)  
        np.save(os.path.join(folder_path,'apic_v_array.npy'), self.apic_v_array)
        np.save(os.path.join(folder_path,'apic_ica_array.npy'), self.apic_ica_array)

        np.save(os.path.join(folder_path,'dend_v_array.npy'), self.dend_v_array)
        np.save(os.path.join(folder_path,'dend_i_array.npy'), self.dend_i_array)
        np.save(os.path.join(folder_path,'dend_nmda_i_array.npy'), self.dend_nmda_i_array)
        np.save(os.path.join(folder_path,'dend_ampa_i_array.npy'), self.dend_ampa_i_array)
        
    def run_simulation(self, num_aff_fiber, num_trial):

        soma_v = h.Vector().record(self.complex_cell.soma[0](0.5)._ref_v)
        apic_v = h.Vector().record(self.complex_cell.apic[36](1)._ref_v)
        apic_ica = h.Vector().record(self.complex_cell.apic[36](1)._ref_ica)

        # Record summed local background NMDA current
        bg_exc_i_nmda_list = []
        bg_exc_i_ampa_list = []

        for exc_syn in exc_syn_on_clustered_sec['synapse']:
            bg_exc_i_nmda = h.Vector().exc_syn.record(exc_syn._ref_i_NMDA)
            bg_exc_i_ampa = h.Vector().exc_syn.record(exc_syn._ref_i_AMPA)

            bg_exc_i_nmda_list.append(bg_exc_i_nmda)
            bg_exc_i_ampa_list.append(bg_exc_i_ampa)

        # 创建用于保存所有 cluster 记录的列表
        dend_v_list = []
        dend_i_list = []
        dend_i_nmda_list = []
        dend_i_ampa_list = []

        # 假设 clusters 是包含所有 cluster 的列表或数组
        for cluster_id in range(self.num_clusters):
            # 获取当前 cluster 的 basal_ctr 和 syn_ctr
            cluster_basal_ctr = self.section_synapse_df[self.section_synapse_df['cluster_id'] == cluster_id]['segment_synapse'].values[0]
            # print('cluster_basal_ctr: ', cluster_basal_ctr)
            syn_ctr = self.section_synapse_df[self.section_synapse_df['cluster_id'] == cluster_id]['synapse'].values[0]
            
            # 创建新的 dend_v 和 dend_i，并记录相应的电压和电流
            dend_v = h.Vector().record(cluster_basal_ctr._ref_v)
            dend_i = h.Vector().record(syn_ctr._ref_i)
            
            try:
                dend_i_nmda = h.Vector().record(syn_ctr._ref_i_NMDA)
            except AttributeError:
                dend_i_nmda = h.Vector().record(syn_ctr._ref_i_AMPA)
            
            dend_i_ampa = h.Vector().record(syn_ctr._ref_i_AMPA)

            # 将 dend_v 和 dend_i 添加到列表中
            dend_v_list.append(dend_v)
            dend_i_list.append(dend_i)
            dend_i_nmda_list.append(dend_i_nmda)
            dend_i_ampa_list.append(dend_i_ampa)

        h.tstop = self.DURATION
        h.run()
              
        self.soma_v_array[:, num_aff_fiber, num_trial] = np.array(soma_v)
        self.apic_v_array[:, num_aff_fiber, num_trial] = np.array(apic_v)
        self.apic_ica_array[:, num_aff_fiber, num_trial] = np.array(apic_ica)

        for cluster_id in range(self.num_clusters):
            self.dend_v_array[cluster_id, :, num_aff_fiber, num_trial] = np.array(dend_v_list[cluster_id])
            self.dend_i_array[cluster_id, :, num_aff_fiber, num_trial] = np.array(dend_i_list[cluster_id])
            self.dend_nmda_i_array[cluster_id, :, num_aff_fiber, num_trial] = np.array(dend_i_nmda_list[cluster_id])
            self.dend_ampa_i_array[cluster_id, :, num_aff_fiber, num_trial] = np.array(dend_i_ampa_list[cluster_id])
        
    def add_single_synapse(self, num_syn, region, sim_type):
        sections = self.sections_basal if region == 'basal' else self.sections_apical
        # e_syn, tau1, tau2, syn_weight = self.syn_param_exc if sim_type == 'exc' else self.syn_param_inh
        type = 'A' if sim_type == 'exc' else 'B'
        
        if region == 'basal':
            section_length = np.array(self.section_df.loc[self.section_df['section_type'] == 'dend', 'length'])  
        else:
            section_length = np.array(self.section_df.loc[self.section_df['section_type'] == 'apic', 'length'])

        def generate_synapse(_):
            section = random.choices(sections, weights=section_length)[0][0].sec
            section_name = section.psection()['name']
            
            section_id_synapse = self.section_df.loc[self.section_df['section_name'] == section_name, 'section_id'].values[0]

            loc = self.rnd.uniform()
            segment_synapse = section(loc)

            data_to_append = {'section_id_synapse': section_id_synapse,
                            'section_synapse': section,
                            'segment_synapse': segment_synapse,
                            'synapse': None, 
                            'netstim': None,
                            'random': None,
                            'netcon': None,
                            'loc': loc,
                            'type': type,
                            'cluster_center_flag': -1,
                            'cluster_id': -1,
                            'pre_unit_id': -1,
                            'region': region}

            with self.lock:
                self.section_synapse_df = self.section_synapse_df.append(data_to_append, ignore_index=True)

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            list(tqdm(executor.map(generate_synapse, range(num_syn)), total=num_syn))
 

# main function
import sys 
import json
import multiprocessing
from utils.genarate_simu_params_utils import generate_simu_params
sys.setrecursionlimit(1000000)

# swc_file_path = './modelFile/cell1.asc'
swc_file_path = '/home/mimo/Desktop/neuron_reduce/example/modelFile/cell1.asc'

def build_cell(**params):

    NUM_SYN_BASAL_EXC, \
    NUM_SYN_APIC_EXC, \
    NUM_SYN_BASAL_INH, \
    NUM_SYN_APIC_INH, \
    DURATION, \
    basal_channel_type, \
    sec_type, \
    distance_to_root, \
    num_clusters, \
    cluster_radius, \
    bg_exc_freq, \
    bg_inh_freq, \
    bg_exc_channel_type, \
    initW, \
    inh_delay, \
    num_stim, \
    num_conn_per_preunit, \
    num_preunit, \
    pref_ori_dg, \
    num_trials, \
    folder_tag = params.values()

    # 创建保存文件夹
    time_tag = time.strftime("%Y%m%d_%H%M", time.localtime())

    # folder_path = './results/simulation/pseudo/' + basal_channel_type + '_' + time_tag + '/' + folder_tag
    folder_path = 'D:/results/simulation/pseudo/' + time_tag + '/' + folder_tag

    simulation_params = {
        'cell model': 'L5PN',
        'NUM_SYN_BASAL_EXC': NUM_SYN_BASAL_EXC,
        'NUM_SYN_APIC_EXC': NUM_SYN_APIC_EXC,
        'NUM_SYN_BASAL_INH': NUM_SYN_BASAL_INH,
        'NUM_SYN_APIC_INH': NUM_SYN_APIC_INH,
        'DURATION': DURATION,
        'basal channel type': basal_channel_type,
        'section type': sec_type,
        'distance from basal clusters to root': distance_to_root,
        'number of clusters': num_clusters,
        'cluster radius': cluster_radius,
        'background excitatory frequency': bg_exc_freq,
        'background inhibitory frequency': bg_inh_freq,
        'background excitatory channel type': bg_exc_channel_type,
        'initial weight of AMPANMDA synapses': initW,
        'delay of inhibitory inputs': inh_delay,
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

    cell1 = CellWithNetworkx(swc_file_path, bg_exc_freq, bg_inh_freq, DURATION)
    cell1.add_synapses(NUM_SYN_BASAL_EXC, 
                                NUM_SYN_APIC_EXC, 
                                NUM_SYN_BASAL_INH, 
                                NUM_SYN_APIC_INH)
    
    cell1.assign_clustered_synapses(basal_channel_type, sec_type,
                                    distance_to_root, num_clusters, 
                                    cluster_radius, num_stim, 
                                    num_conn_per_preunit, num_preunit,
                                    folder_path) 

    cell1.add_inputs(folder_path, bg_exc_channel_type, initW, inh_delay, num_trials)

def run_processes(parameters_list):

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