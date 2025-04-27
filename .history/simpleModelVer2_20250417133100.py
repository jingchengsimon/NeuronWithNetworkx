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

class CellWithNetworkx:
    def __init__(self, swc_file, bg_exc_freq, bg_inh_freq, SIMU_DURATION, STIM_DURATION, epoch_idx):
        h.load_file("import3d.hoc")

        # h.nrn_load_dll('./mod/nrnmech.dll') # For Windows
        h.nrn_load_dll('./mod/x86_64/.libs/libnrnmech.so') # For Linux/Mac
        h.load_file('./modelFile/L5PCbiophys3.hoc')
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

        self.rnd = np.random.default_rng(epoch_idx) #np.random.RandomState(42) 
        random.seed(epoch_idx) 
        self.epoch_idx = epoch_idx # random seed for the current epoch

        if bg_exc_freq != 0:
            self.spike_interval = 1000/bg_exc_freq # interval=1000(ms)/f
        self.FREQ_EXC = bg_exc_freq  # Hz, /s
        self.FREQ_INH = bg_inh_freq  # Hz, /s
        self.SIMU_DURATION = SIMU_DURATION # 1s 
        self.STIM_DURATION = STIM_DURATION # 1s

        self.syn_param_exc = [0, 0.3, 1.8] # reversal_potential, tau1, tau2, syn_weight (actually we don't use these params, delete later)
        self.syn_param_inh = [-86, 1, 8, 0.00069] #->0.00069 uS = 0.69 nS

        self.sections_soma = [i for i in map(list, list(self.complex_cell.soma))]
        self.sections_basal = [i for i in map(list, list(self.complex_cell.basal))] 
        self.sections_apical = [i for i in map(list, list(self.complex_cell.apical))]
        self.all_sections = self.sections_soma + self.sections_basal + self.sections_apical   
                                       
        self.section_synapse_df = pd.DataFrame(columns=['section_id_synapse',
                                                'section_synapse',
                                                'segment_synapse',
                                                'synapse', 
                                                'netstim',  
                                                'netcon',
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
                                                'spike_train',
                                                'spike_train_bg',
                                                'syn_w'], dtype=object) # for adding vecstim of different orientation
                                        
        # For clustered synapses
        self.basal_channel_type = None
        self.sec_type = None
        self.distance_to_soma = None
        self.num_clusters = None
        self.cluster_radius = None

        self.num_clusters_sampled = None

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
        self.apic_ica_array = None

        self.trunk_v_array = None
        self.basal_v_array = None
        self.tuft_v_array = None

        self.basal_bg_i_nmda_array = None
        self.basal_bg_i_ampa_array = None
        self.tuft_bg_i_nmda_array = None
        self.tuft_bg_i_ampa_array = None

        self.dend_v_array = None
        self.dend_i_array = None
        self.dend_nmda_i_array = None
        self.dend_ampa_i_array = None
        self.dend_nmda_g_array = None
        self.dend_ampa_g_array = None

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
        self.section_df, self.DiG = create_directed_graph(self.all_sections, self.section_df)

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
                           
    def assign_clustered_synapses(self, basal_channel_type, sec_type, dis_to_root, 
                                  num_clusters, cluster_radius, num_stim, stim_time, 
                                  spat_condition, num_conn_per_preunit, num_syn_per_clus,
                                  folder_path):
        
        # self.section_synapse_df.to_csv(os.path.join(folder_path, 'section_synapse_df.csv'), index=False)
        
        # Extract distances
        basal_distance = self.section_synapse_df[
            (self.section_synapse_df['region'] == 'basal') & 
            (self.section_synapse_df['type'] == 'A')]['distance_to_soma'].values
        tuft_distance = self.section_synapse_df[
            (self.section_synapse_df['distance_to_tuft'] != -1) & 
            (self.section_synapse_df['type'] == 'A')]['distance_to_tuft'].values

        # Sort the distances
        sorted_basal_distances = np.sort(basal_distance)
        sorted_tuft_distances = np.sort(tuft_distance)

        if sec_type == 'basal':
            num_syn_thres = [3000 + i * 3000 for i in range(2)]
        elif sec_type == 'apical':
            num_syn_thres = [2500 + i * 2500 for i in range(2)]

        # Get the indices for the thresholds
        dist_thres_basal = [0] + [sorted_basal_distances[threshold - 1] for threshold in num_syn_thres 
                                  if threshold <= len(sorted_basal_distances)] + [max(sorted_basal_distances)]

        dist_thres_tuft = [0] + [sorted_tuft_distances[threshold - 1] for threshold in num_syn_thres 
                                 if threshold <= len(sorted_tuft_distances)] + [max(sorted_tuft_distances)]

        self.num_clusters_sampled = num_clusters
    
        # Comment only for test
        num_conn_per_preunit = min(num_conn_per_preunit, num_clusters) 
        num_preunit = num_syn_per_clus * np.ceil(num_clusters / 3).astype(int)

        if spat_condition == 'clus':
            # Number of synapses in each cluster is not fixed
            self.pref_ori_dg, indices = generate_indices(self.rnd, num_clusters, 
                                                                    num_conn_per_preunit, num_preunit)
        elif spat_condition == 'distr':
            # num_pre*num_conn clus with 1 syn per 'cluster'
            num_clusters = num_preunit * num_conn_per_preunit
            numbers = np.repeat(np.arange(num_preunit), num_conn_per_preunit)
            self.rnd.shuffle(numbers)
            indices = [[num] for num in numbers]

        self.unit_ids = np.arange(num_preunit)
        self.indices = indices

        # Save assignment
        file_path = os.path.join(folder_path, 'preunit assignment.txt')

        with open(file_path, 'w') as f:
            for i, index_list in enumerate(indices):
                f.write(f"Cluster_id: {i}, Num_preunits: {len(index_list)}, Preunit_ids: {index_list}\n")
        
        self.basal_channel_type = basal_channel_type
        self.sec_type = sec_type
        self.dis_to_root = dis_to_root
        self.num_clusters = num_clusters
        self.cluster_radius = cluster_radius

        self.num_stim = num_stim
        self.stim_time = stim_time
        self.num_conn_per_preunit = num_conn_per_preunit
        self.num_preunit = num_preunit

        for i in range(self.num_clusters):
            # Unassigned background synapses for surround synapses
            sec_syn_bg_exc_df = self.section_synapse_df[(self.section_synapse_df['type'] == 'A') & 
                                                        (self.section_synapse_df['cluster_flag'] == -1)]
                
            # Unassigned background synapses for center synapses
            # Define the concentration level for clus (6 clus on 6 branches / 1 branch)

            basal_branch_idx_list = [40, 41, 41]

            if sec_type == 'basal':
                sec_syn_bg_exc_ordered_df = self.section_synapse_df[
                    (self.section_synapse_df['region'] == sec_type) &
                    (self.section_synapse_df['type'] == 'A') &
                    (self.section_synapse_df['cluster_flag'] == -1)&
                    (self.section_synapse_df['distance_to_soma'].between(dist_thres_basal[dis_to_root], dist_thres_basal[dis_to_root+1]))] # &
                    # (self.section_synapse_df['section_id_synapse'] == basal_branch_idx_list[dis_to_root])] # specifically choose the branch 
                    # (self.section_synapse_df['branch_idx'] == np.random.randint(0, 6))]  # For concentration level: most branchy basal branch_idx: 4; distr_multi: np.random.randint(0, 6)
                
            elif sec_type == 'apical':
                sec_syn_bg_exc_ordered_df = self.section_synapse_df[
                    (self.section_synapse_df['section_id_synapse'].isin(self.sec_tuft_idx)) & # apical includes oblique, trunk and tuft
                    (self.section_synapse_df['type'] == 'A') &
                    (self.section_synapse_df['cluster_flag'] == -1) &
                    (self.section_synapse_df['distance_to_tuft'].between(dist_thres_tuft[dis_to_root], dist_thres_tuft[dis_to_root+1]))]# &
                    # (self.section_synapse_df['loc'] < 0.5) &
                    # (self.section_synapse_df['section_id_synapse'] == apic_branch_idx_list[dis_to_root])] # specifically choose the branch 
                
            index_list = indices[i]
            num_syn_per_clus = len(index_list)  
            
            loop_count = 0
            
            rng = np.random.RandomState(self.epoch_idx)
            # rng = np.random.RandomState(int(time.time()) + loop_count)

            # Loop for cluster assignment
            while True:
                loop_count += 1

                # use the rng for positioning
                syn_ctr = sec_syn_bg_exc_ordered_df.loc[rng.choice(sec_syn_bg_exc_ordered_df.index)]

                # Assign the surround as clustered synapse only if more than 1 syn per cluster (dispersed: 1 syn per cluster)
                if num_syn_per_clus > 1:

                    syn_ctr_sec = syn_ctr['section_synapse']
                    syn_surround_ctr = sec_syn_bg_exc_df[
                        (sec_syn_bg_exc_df['section_synapse'] == syn_ctr_sec) & 
                        (sec_syn_bg_exc_df.index != syn_ctr.name)]

                    dis_syn_from_ctr = np.array(np.abs(syn_ctr['loc'] - syn_surround_ctr['loc']) * syn_ctr_sec.L)
                    # use exponential distribution to generate loc

                    max_num_syn_per_clus = max(num_syn_per_clus, 100)

                    # max_dis_mark_from_ctr = np.sort(self.rnd.exponential(cluster_radius, max_num_syn_per_cluster - 1))
                    try:
                        dis_mark_from_ctr = np.sort(rng.exponential(cluster_radius, num_syn_per_clus - 1))
                        max_dis_mark_from_ctr = np.sort(rng.exponential(cluster_radius, max_num_syn_per_clus - 1))
                    except ValueError:
                        dis_mark_from_ctr = np.sort(rng.exponential(cluster_radius, 0))
                        max_dis_mark_from_ctr = np.sort(rng.exponential(cluster_radius, 0))

                    # not enough synapses on the same section
                    syn_ctr_sec_id = syn_ctr['section_id_synapse']
                    syn_suc_sec_id = syn_ctr_sec_id
                    syn_pre_sec_id = syn_ctr_sec_id
                    
                    exceed_flag = False

                    while len(dis_syn_from_ctr) < max_num_syn_per_clus - 1:
                    # while len(dis_syn_from_ctr) < num_syn_per_clus - 1:
                
                        # Check and empty syn_pre_surround_ctr and syn_suc_surround_ctr if they exist
                        if 'syn_pre_surround_ctr' in locals():
                            syn_pre_surround_ctr = syn_pre_surround_ctr.iloc[0:0]

                        if 'syn_suc_surround_ctr' in locals():
                            syn_suc_surround_ctr = syn_suc_surround_ctr.iloc[0:0]

                        # Check and empty dis_syn_pre_from_ctr and dis_syn_suc_from_ctr if they exist
                        if 'dis_syn_pre_from_ctr' in locals():
                            dis_syn_pre_from_ctr = np.array([])

                        if 'dis_syn_suc_from_ctr' in locals():
                            dis_syn_suc_from_ctr = np.array([])

                        # the children section of the center section
                        if list(self.DiG.successors(syn_suc_sec_id)):
                            # iterate
                            syn_suc_sec_id = rng.choice(list(self.DiG.successors(syn_suc_sec_id)))
                            try:
                                syn_suc_sec = sec_syn_bg_exc_df[sec_syn_bg_exc_df['section_id_synapse'] == syn_suc_sec_id]['section_synapse'].values[0]
                                syn_suc_surround_ctr = sec_syn_bg_exc_df[sec_syn_bg_exc_df['section_id_synapse'] == syn_suc_sec_id]
                                dis_syn_suc_from_ctr = np.array((1 - syn_ctr['loc']) * syn_ctr_sec.L + syn_suc_surround_ctr['loc'] * syn_suc_sec.L)
                            except IndexError:
                                # print(f"IndexError: syn_suc_sec_id: {syn_suc_sec_id}")
                                pass

                        # the parent section of the center section
                        # there is no dendritic section on the soma, so we should not choose soma as the parent section
                        # also don't choose the apical nexus section as the parent section
                        if list(self.DiG.predecessors(syn_pre_sec_id)) not in ([], [0], [121]):
                            syn_pre_sec_id = rng.choice(list(self.DiG.predecessors(syn_pre_sec_id)))
                            try:
                                syn_pre_sec = sec_syn_bg_exc_df[sec_syn_bg_exc_df['section_id_synapse'] == syn_pre_sec_id]['section_synapse'].values[0]
                                syn_pre_surround_ctr = sec_syn_bg_exc_df[sec_syn_bg_exc_df['section_id_synapse'] == syn_pre_sec_id]
                                dis_syn_pre_from_ctr = np.array(syn_ctr['loc'] * syn_ctr_sec.L + (1 - syn_pre_surround_ctr['loc']) * syn_pre_sec.L)
                            except IndexError:
                                # print(f"IndexError: syn_pre_sec_id: {syn_pre_sec_id}")
                                pass
                        
                        # print('ctr:', syn_ctr_sec_id, 'suc:', syn_suc_sec_id, 'pre:', syn_pre_sec_id)

                        arr_to_concat, df_to_concat = [], []

                        # Combine conditions and append to lists
                        for dis_syn, syn_surround in [
                            ('dis_syn_from_ctr', 'syn_surround_ctr'),
                            ('dis_syn_suc_from_ctr', 'syn_suc_surround_ctr'),
                            ('dis_syn_pre_from_ctr', 'syn_pre_surround_ctr')
                        ]:
                            if dis_syn in locals() and syn_surround in locals():
                                arr_to_concat.append(locals()[dis_syn])
                                df_to_concat.append(locals()[syn_surround])

                        # Concatenate arrays and dataframes if not empty
                        if arr_to_concat:
                            dis_syn_from_ctr = np.concatenate(arr_to_concat) 

                        if df_to_concat:
                            syn_surround_ctr = pd.concat(df_to_concat)
                        
                        # unique_dis_syn_from_ctr, unique_indices = np.unique(dis_syn_from_ctr, return_index=True)
                        # dis_syn_from_ctr = unique_dis_syn_from_ctr
                        # syn_surround_ctr = syn_surround_ctr.iloc[unique_indices]

                        # after the loop, if the pre of pre and suc of suc exceed the sec_id of the sec_syn_bg_exc_ordered_df but the len(dis_syn_from_ctr) still does not reach the standard,
                        # break the loop and re-choose the syn_ctr (the chosen one be reset to type 'A')
                        suc_exceed_flag = (list(self.DiG.successors(syn_suc_sec_id)) == []) or (not any(sec_id in np.unique(sec_syn_bg_exc_ordered_df['section_id_synapse']) 
                                                                                                        for sec_id in list(self.DiG.successors(syn_suc_sec_id))))
                        pre_exceed_flag = not any(sec_id in np.unique(sec_syn_bg_exc_ordered_df['section_id_synapse']) 
                                                  for sec_id in list(self.DiG.predecessors(syn_pre_sec_id)))
                        exceed_flag = suc_exceed_flag and pre_exceed_flag and (len(dis_syn_from_ctr) < max_num_syn_per_clus - 1)
                        
                        # print('suc_flag:', suc_exceed_flag, 'pre_flag:', pre_exceed_flag, 'exceed_flag:', exceed_flag)

                        if exceed_flag:
                            break

                    if exceed_flag:
                        continue
                    
                    max_clus_mem_idx = distance_synapse_mark_compare(dis_syn_from_ctr, max_dis_mark_from_ctr)
                    clus_mem_idx = rng.choice(max_clus_mem_idx, num_syn_per_clus - 1, replace=False)

                    # assign the surround as clustered synapse
                    self.section_synapse_df.loc[syn_surround_ctr.iloc[clus_mem_idx].index, 'cluster_flag'] = 1
                    self.section_synapse_df.loc[syn_surround_ctr.iloc[clus_mem_idx].index, 'cluster_center_flag'] = 0
                    self.section_synapse_df.loc[syn_surround_ctr.iloc[clus_mem_idx].index, 'cluster_id'] = i
                    for j in range(len(clus_mem_idx)):
                        try:
                            self.section_synapse_df.loc[syn_surround_ctr.iloc[clus_mem_idx].index[j], 'pre_unit_id'] = index_list[j+1]
                        except IndexError:
                            self.section_synapse_df.loc[syn_surround_ctr.iloc[clus_mem_idx].index[j], 'pre_unit_id'] = -1         
                break

            # assign the center as clustered synapse
            self.section_synapse_df.loc[syn_ctr.name, 'cluster_flag'] = 1
            self.section_synapse_df.loc[syn_ctr.name, 'cluster_center_flag'] = 1
            self.section_synapse_df.loc[syn_ctr.name, 'cluster_id'] = i
            try:
                self.section_synapse_df.loc[syn_ctr.name, 'pre_unit_id'] = index_list[0]
            except IndexError:
                self.section_synapse_df.loc[syn_ctr.name, 'pre_unit_id'] = -1

            if i < 10:
                if num_syn_per_clus > 1:
                    print(np.unique(syn_surround_ctr['section_id_synapse']))
                else:
                    print(np.unique(syn_ctr['section_id_synapse']))
                    
            # print('cluster_id:', i, len(dis_syn_from_ctr), len(clus_mem_idx))
            # print('num_syn_per_clus: ', len(self.section_synapse_df[(self.section_synapse_df['cluster_id'] == i)]['segment_synapse'].values))
        
            # print('next')
                
    def add_inputs(self, folder_path, simu_condition, input_ratio_basal_apic, bg_exc_channel_type, initW, num_func_group, inh_delay, num_trials):
        
        self.input_ratio_basal_apic = input_ratio_basal_apic
        self.bg_exc_channel_type = bg_exc_channel_type
        self.initW = initW
        self.num_func_group = num_func_group
        self.inh_delay = inh_delay

        spt_unit_array_list = []
        for num_stim in range(1, self.num_stim + 1):
            spt_unit_array = generate_vecstim(self.unit_ids, num_stim, self.stim_time)
            spt_unit_array_list.append(spt_unit_array)
        
        self.num_syn_inh_list = [self.num_syn_basal_inh, self.num_syn_apic_inh, self.num_syn_soma_inh]
        
        # create an ndarray to store the voltage of each cluster of each trial 
        num_time_points = 1 + 40 * self.SIMU_DURATION
        
        iter_step = 2
        self.num_activated_preunit_list = range(0, self.num_preunit + 1, iter_step) # for varynumsyn (add 1 is to allow the last num_preunit to be included)
        # self.num_activated_preunit_list = [self.num_preunit] # for heatmap
        num_aff_fibers = len(self.num_activated_preunit_list)

        perm = np.random.permutation(self.num_preunit)

        self.soma_v_array = np.zeros((num_time_points, self.num_stim, num_aff_fibers, num_trials))
        self.apic_v_array = np.zeros((num_time_points, self.num_stim, num_aff_fibers, num_trials))
        self.apic_ica_array = np.zeros((num_time_points, self.num_stim, num_aff_fibers, num_trials))

        self.soma_i_array = np.zeros((num_time_points, self.num_stim, num_aff_fibers, num_trials))

        self.trunk_v_array = np.zeros((num_time_points, self.num_stim, num_aff_fibers, num_trials))
        self.basal_v_array = np.zeros((num_time_points, self.num_stim, num_aff_fibers, num_trials))
        self.tuft_v_array = np.zeros((num_time_points, self.num_stim, num_aff_fibers, num_trials))

        self.basal_bg_i_nmda_array = np.zeros((num_time_points, self.num_stim, num_aff_fibers, num_trials))
        self.basal_bg_i_ampa_array = np.zeros((num_time_points, self.num_stim, num_aff_fibers, num_trials))
        self.tuft_bg_i_nmda_array = np.zeros((num_time_points, self.num_stim, num_aff_fibers, num_trials))
        self.tuft_bg_i_ampa_array = np.zeros((num_time_points, self.num_stim, num_aff_fibers, num_trials))

        self.dend_v_array = np.zeros((self.num_clusters_sampled, num_time_points, self.num_stim, num_aff_fibers, num_trials))
        self.dend_i_array = np.zeros((self.num_clusters_sampled, num_time_points, self.num_stim, num_aff_fibers, num_trials))
        self.dend_nmda_i_array = np.zeros((self.num_clusters_sampled, num_time_points, self.num_stim, num_aff_fibers, num_trials))
        self.dend_ampa_i_array = np.zeros((self.num_clusters_sampled, num_time_points, self.num_stim, num_aff_fibers, num_trials))
        self.dend_nmda_g_array = np.zeros((self.num_clusters_sampled, num_time_points, self.num_stim, num_aff_fibers, num_trials))
        self.dend_ampa_g_array = np.zeros((self.num_clusters_sampled, num_time_points, self.num_stim, num_aff_fibers, num_trials)) 

        # condition_met = False  # Flag to indicate if the condition has been met
        
        if 'distr' in folder_path:
            spat_condition = 'distr'
            forlder_path_clus = folder_path.replace('distr', 'clus')
            section_synapse_df_clus = pd.read_csv(os.path.join(forlder_path_clus, 'section_synapse_df.csv'))
        else:
            spat_condition = 'clus'
            section_synapse_df_clus = self.section_synapse_df

        if simu_condition == 'invivo':
                    
            add_background_exc_inputs(self.section_synapse_df, 
                                    self.syn_param_exc, 
                                    self.SIMU_DURATION,
                                    self.FREQ_EXC, 
                                    self.input_ratio_basal_apic,
                                    self.bg_exc_channel_type,
                                    self.initW,
                                    self.num_func_group,
                                    self.rnd,
                                    self.epoch_idx,
                                    spat_condition,
                                    section_synapse_df_clus,
                                    self.lock)
        
        for num_activated_preunit in self.num_activated_preunit_list:  

            # if condition_met:
            #     break  # End the whole loop if the condition has been met

            for num_stim in range(self.num_stim):
                for num_trial in range(num_trials): # 20

                    # if simu_condition == 'invivo':
                    
                    #     add_background_exc_inputs(self.section_synapse_df, 
                    #                             self.syn_param_exc, 
                    #                             self.SIMU_DURATION,
                    #                             self.FREQ_EXC, 
                    #                             self.input_ratio_basal_apic,
                    #                             self.bg_exc_channel_type,
                    #                             self.initW,
                    #                             self.num_func_group,
                    #                             self.rnd,
                    #                             self.epoch_idx,
                    #                             spat_condition,
                    #                             section_synapse_df_clus,
                    #                             self.lock)
                    
                    # spt_unit_list_list = []
                    # for num_stim_idx in range(1, self.num_stim + 1):
                    #     spt_unit_list = generate_vecstim(self.unit_ids, num_stim_idx, self.stim_time)
                    #     spt_unit_list_list.append(spt_unit_list)

                    spt_unit_array = spt_unit_array_list[num_stim]
                    spt_unit_array_truncated = spt_unit_array[perm[:num_activated_preunit]]
                    # spt_unit_list_truncated = random.sample(spt_unit_list, num_activated_preunit)
                    # spt_unit_list_truncated = spt_unit_list
                    
                    add_clustered_inputs(self.section_synapse_df, 
                                        self.num_clusters, 
                                        self.basal_channel_type,   
                                        self.initW,
                                        spt_unit_array_truncated,
                                        self.rnd)
                    
                # for num_trial in range(num_trials): # 20

                    # Add background inputs for in vivo-like condition
                    if simu_condition == 'invivo':
                        
                        add_background_inh_inputs(self.section_synapse_df, 
                                                self.syn_param_inh, 
                                                self.SIMU_DURATION, 
                                                self.FREQ_INH, 
                                                self.input_ratio_basal_apic,
                                                self.num_syn_inh_list, 
                                                self.inh_delay,
                                                self.rnd,
                                                self.epoch_idx,
                                                spat_condition,
                                                section_synapse_df_clus,
                                                self.lock)
                
                    # Run the simulation
                    num_aff_idx = self.num_activated_preunit_list.index(num_activated_preunit)

                    self.run_simulation(num_stim, num_aff_idx, num_trial, folder_path)

                    # if not self.run_simulation(num_stim, num_aff_idx, num_trial):
                    #     break  # Skip to the next epoch if the condition is not met
                    # condition_met = True
                    # print(f"Met for numpreunit={num_activated_preunit}")

        np.save(os.path.join(folder_path,'soma_v_array.npy'), self.soma_v_array)  
        np.save(os.path.join(folder_path,'apic_v_array.npy'), self.apic_v_array)
        np.save(os.path.join(folder_path,'apic_ica_array.npy'), self.apic_ica_array)
        
        np.save(os.path.join(folder_path,'soma_i_array.npy'), self.soma_i_array)

        np.save(os.path.join(folder_path,'trunk_v_array.npy'), self.trunk_v_array)
        np.save(os.path.join(folder_path,'basal_v_array.npy'), self.basal_v_array)
        np.save(os.path.join(folder_path,'tuft_v_array.npy'), self.tuft_v_array)

        np.save(os.path.join(folder_path,'basal_bg_i_nmda_array.npy'), self.basal_bg_i_nmda_array)
        np.save(os.path.join(folder_path,'basal_bg_i_ampa_array.npy'), self.basal_bg_i_ampa_array)
        np.save(os.path.join(folder_path,'tuft_bg_i_nmda_array.npy'), self.tuft_bg_i_nmda_array)
        np.save(os.path.join(folder_path,'tuft_bg_i_ampa_array.npy'), self.tuft_bg_i_ampa_array)

        np.save(os.path.join(folder_path,'dend_v_array.npy'), self.dend_v_array)
        np.save(os.path.join(folder_path,'dend_i_array.npy'), self.dend_i_array)
        np.save(os.path.join(folder_path,'dend_nmda_i_array.npy'), self.dend_nmda_i_array)
        np.save(os.path.join(folder_path,'dend_ampa_i_array.npy'), self.dend_ampa_i_array)
        np.save(os.path.join(folder_path,'dend_nmda_g_array.npy'), self.dend_nmda_g_array)
        np.save(os.path.join(folder_path,'dend_ampa_g_array.npy'), self.dend_ampa_g_array)

        self.section_synapse_df.to_csv(os.path.join(folder_path, 'section_synapse_df.csv'), index=False)
        
    def run_simulation(self, num_stim, num_aff_fiber, num_trial, folder_path):

        soma_v = h.Vector().record(self.complex_cell.soma[0](0.5)._ref_v)
        apic_v = h.Vector().record(self.complex_cell.apic[121-85](1)._ref_v)
        apic_ica = h.Vector().record(self.complex_cell.apic[121-85](1)._ref_ica)

        trunk_v = h.Vector().record(self.complex_cell.apic[3](0)._ref_v)
        basal_v = h.Vector().record(self.complex_cell.dend[71-1](0.5)._ref_v) # the 71th dendrite (tip), L: 178.7, order: 3, distance to root: 192.8
        tuft_v = h.Vector().record(self.complex_cell.apic[152-85](0.5)._ref_v) # the 152th dendrite (tip), L: 192.8, order: 3, distance to root: 565.0

        # EPSC record (VClamp)
        vc = h.SEClamp(self.complex_cell.soma[0](0.5))   
        # vc.dur1 = 1000  # Long duration to hold the voltage
        # vc.amp1 = 60   # Holding voltage at 60 mV
        soma_i = h.Vector().record(vc._ref_i)

        try:
            # Record summed local background NMDA current at the basal tip branch
            exc_syn_on_basal_sec = self.section_synapse_df[(self.section_synapse_df['section_id_synapse'] == 71) &
                                                        (self.section_synapse_df['type'] == 'A')]['synapse']
            basal_bg_i_nmda_list = []
            basal_bg_i_ampa_list = []
            
            for exc_syn in exc_syn_on_basal_sec:

                try:
                    basal_bg_i_nmda = h.Vector().record(exc_syn._ref_i_NMDA)
                except AttributeError:
                    basal_bg_i_nmda = h.Vector().record(exc_syn._ref_i_AMPA)

                basal_bg_i_ampa = h.Vector().record(exc_syn._ref_i_AMPA)

                basal_bg_i_nmda_list.append(basal_bg_i_nmda)
                basal_bg_i_ampa_list.append(basal_bg_i_ampa)

            # Record summed local background NMDA current at the tuft tip branch
            exc_syn_on_tuft_sec = self.section_synapse_df[(self.section_synapse_df['section_id_synapse'] == 152) &
                                                        (self.section_synapse_df['type'] == 'A')]['synapse']
            tuft_bg_i_nmda_list = []  
            tuft_bg_i_ampa_list = []

            for exc_syn in exc_syn_on_tuft_sec:

                try:
                    tuft_bg_i_nmda = h.Vector().record(exc_syn._ref_i_NMDA)                
                except AttributeError:
                    tuft_bg_i_nmda = h.Vector().record(exc_syn._ref_i_AMPA)

                tuft_bg_i_ampa = h.Vector().record(exc_syn._ref_i_AMPA)

                tuft_bg_i_nmda_list.append(tuft_bg_i_nmda)
                tuft_bg_i_ampa_list.append(tuft_bg_i_ampa)
            
        except AttributeError:
            pass

        # Record center synapse voltage and current at each cluster
        dend_v_list = []
        dend_i_list_list = []
        dend_i_nmda_list_list = []
        dend_i_ampa_list_list = []
        dend_g_nmda_list_list = []
        dend_g_ampa_list_list = []

        print('num_syn_per_clus: ', [len(self.section_synapse_df[(self.section_synapse_df['cluster_id'] == i)]['segment_synapse'].values) for i in range(self.num_clusters_sampled)],
              ' num_clus: ', len(self.section_synapse_df[(self.section_synapse_df['cluster_center_flag'] == 1)]['cluster_id'].values), '\n')
              
        for cluster_id in range(self.num_clusters_sampled):

            # choose the center synapse of each cluster
            cluster_ctr = self.section_synapse_df[(self.section_synapse_df['cluster_id'] == cluster_id) &
                                                  (self.section_synapse_df['cluster_center_flag'] == 1)]['segment_synapse'].values[0]
            
            dend_v = h.Vector().record(cluster_ctr._ref_v)

            clustered_sec = np.unique(self.section_synapse_df[self.section_synapse_df['cluster_id'] == cluster_id]['section_synapse'])
            exc_syn_on_clus_sec = self.section_synapse_df[(self.section_synapse_df['section_synapse'].isin(clustered_sec)) & 
                                                            (self.section_synapse_df['type'].isin(['A']))]['synapse']
            exc_syn_on_clus_sec_filt = list(filter(None, exc_syn_on_clus_sec)) # Not work: exc_syn_on_clus_sec[exc_syn_on_clus_sec!=None]
            
            dend_i_list = []
            dend_i_nmda_list = []
            dend_i_ampa_list = []
            dend_g_nmda_list = []
            dend_g_ampa_list = []

            for exc_syn in exc_syn_on_clus_sec_filt:
                
                dend_i = h.Vector().record(exc_syn._ref_i)

                try:
                    dend_i_nmda = h.Vector().record(exc_syn._ref_i_NMDA)
                    dend_g_nmda = h.Vector().record(exc_syn._ref_g_NMDA)
                except AttributeError:
                    dend_i_nmda = h.Vector().record(exc_syn._ref_i_AMPA)
                    dend_g_nmda = h.Vector().record(exc_syn._ref_g_AMPA)

                dend_i_ampa = h.Vector().record(exc_syn._ref_i_AMPA)
                dend_g_ampa = h.Vector().record(exc_syn._ref_g_AMPA)
                
                dend_i_list.append(dend_i)
                dend_i_nmda_list.append(dend_i_nmda)
                dend_i_ampa_list.append(dend_i_ampa)
                dend_g_nmda_list.append(dend_g_nmda)
                dend_g_ampa_list.append(dend_g_ampa)

            dend_v_list.append(dend_v)
            dend_i_list_list.append(dend_i_list)
            dend_i_nmda_list_list.append(dend_i_nmda_list)
            dend_i_ampa_list_list.append(dend_i_ampa_list)
            dend_g_nmda_list_list.append(dend_g_nmda_list)
            dend_g_ampa_list_list.append(dend_g_ampa_list)

        # Reset the voltage of segments
        seg_v = [h.Vector().record(seg._ref_v) for sec in h.allsec() for seg in sec]

        # netcons_list = list(self.section_synapse_df[(self.section_synapse_df['type'] == 'B')]['netcon'].values[:3])
        # spk_trains_list = list(self.section_synapse_df[(self.section_synapse_df['type'] == 'B')]['spike_train'].values[:3])

        # spike_times = [h.Vector() for _ in netcons_list]
        # for nc, spike_times_vec in zip(netcons_list, spike_times):
        #     nc.record(spike_times_vec)

        # Simulate the full neuron for 1 seconds
        time_start = time.time()
        h.tstop = self.SIMU_DURATION
        h.run()
        print(f"Simulation time: {np.round(time.time() - time_start, 2)}")
        
        # for i in range(len(spike_times)):
        #     try:
        #         print(np.array(spike_times[i]))
        #     except ValueError:
        #         print([])

        #     try:
        #         print(spk_trains_list[i])
        #     except ValueError:
        #         print([])

        # if np.array(soma_v).max() < 0:
        #     return False

        visualize_morpho(self.section_synapse_df, soma_v, seg_v, folder_path)

        with self.lock:

            self.soma_v_array[:, num_stim, num_aff_fiber, num_trial] = np.array(soma_v)
            self.apic_v_array[:, num_stim, num_aff_fiber, num_trial] = np.array(apic_v)
            self.apic_ica_array[:, num_stim, num_aff_fiber, num_trial] = np.array(apic_ica)

            self.soma_i_array[:, num_stim, num_aff_fiber, num_trial] = np.array(soma_i)

            self.trunk_v_array[:, num_stim, num_aff_fiber, num_trial] = np.array(trunk_v)
            self.basal_v_array[:, num_stim, num_aff_fiber, num_trial] = np.array(basal_v)
            self.tuft_v_array[:, num_stim, num_aff_fiber, num_trial] = np.array(tuft_v)

            try:
                self.basal_bg_i_nmda_array[:, num_stim, num_aff_fiber, num_trial] = np.average(np.array(basal_bg_i_nmda_list), axis=0)
                self.basal_bg_i_ampa_array[:, num_stim, num_aff_fiber, num_trial] = np.average(np.array(basal_bg_i_ampa_list), axis=0)
                self.tuft_bg_i_ampa_array[:, num_stim, num_aff_fiber, num_trial] = np.average(np.array(tuft_bg_i_ampa_list), axis=0)
                self.tuft_bg_i_nmda_array[:, num_stim, num_aff_fiber, num_trial] = np.average(np.array(tuft_bg_i_nmda_list), axis=0)

            except UnboundLocalError:
                pass
            
            for cluster_id in range(self.num_clusters_sampled):
                self.dend_v_array[cluster_id, :, num_stim, num_aff_fiber, num_trial] = np.array(dend_v_list[cluster_id])
                self.dend_i_array[cluster_id, :, num_stim, num_aff_fiber, num_trial] = np.average(np.array(dend_i_list_list[cluster_id]), axis=0)
                self.dend_nmda_i_array[cluster_id, :, num_stim, num_aff_fiber, num_trial] = np.average(np.array(dend_i_nmda_list_list[cluster_id]), axis=0)
                self.dend_nmda_g_array[cluster_id, :, num_stim, num_aff_fiber, num_trial] = np.average(np.array(dend_g_nmda_list_list[cluster_id]), axis=0)
                
                self.dend_ampa_i_array[cluster_id, :, num_stim, num_aff_fiber, num_trial] = np.average(np.array(dend_i_ampa_list_list[cluster_id]), axis=0)
                self.dend_ampa_g_array[cluster_id, :, num_stim, num_aff_fiber, num_trial] = np.average(np.array(dend_g_ampa_list_list[cluster_id]), axis=0)

        return True
    
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
                            'synapse': None, 
                            'netstim': None,
                            'netcon': None,
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
                            'spike_train': [],
                            'spike_train_bg': [],
                            'syn_w': None,}

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
    epoch= params.values()

    # time_tag = time.strftime("%Y%m%d", time.localtime())
    # folder_path = '/G/results/simulation/' + time_tag + '/' + folder_tag

    simu_folder = sec_type + '_range' + str(distance_to_root) + '_' + spat_condtion + '_' + simu_condition + '_variedW_tau43_addNaK_nonlintest_varynumsyn_step2_woAP+Ca_withclus_aligned_test' # + '_ratio1' + '_exc1.1-1.3' + '_inh4' + '_failprob0.5' + '_funcgroup10'
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
    sec_type, spat_cond, dis_to_root, epoch = args
    params_list = generate_simu_params(sec_type, spat_cond, dis_to_root)
    # for epoch in range(1, 2):
    run_processes(params_list, epoch)

if __name__ == "__main__":

    ## Running for nonlinearity 
    combinations = [
        (sec_type, spat_cond, dis_to_root, epoch)
        for sec_type in ['basal']
        for spat_cond in ['clus']
        for dis_to_root in [0, 1, 2] 
        for epoch in range(1, 16)
    ]
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:  # 根据CPU核心数调整 multiprocessing.cpu_count()
        executor.map(run_combination, combinations)



    # multiprocessing.set_start_method('spawn', force=True) # Use spawn will initiate too many NEURON instances 

    # # Running for heatmap 
    # for sec_type in ['basal']: # ['basal', 'apical']
    #     for spat_cond in ['clus', 'distr']: # ['clus', 'distr']
    #         for dis_to_root in [0]: # [0, 1, 2]
    #             params_list = generate_simu_params(sec_type, spat_cond, dis_to_root)
    #             for epoch in range(1, 31):
    #                 run_processes(params_list, epoch)

