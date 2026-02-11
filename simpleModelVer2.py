from neuron import gui, h
from pathlib import Path
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
import warnings
import random

import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing
import itertools

import sys 
import json
import argparse
from collections import defaultdict

from utils.graph_utils import create_directed_graph, set_graph_order
from utils.add_inputs_utils import add_background_exc_inputs, add_background_inh_inputs, add_clustered_inputs
from utils.distance_utils import distance_synapse_mark_compare, recur_dist_to_soma, recur_dist_to_root
from utils.generate_stim_utils import generate_indices, generate_vecstim

from utils.visualize_utils import visualize_synapses

sys.setrecursionlimit(1000000)
sys.path.insert(0, '/G/MIMOlab/Codes/NeuronWithNetworkx/mod')

warnings.simplefilter(action='ignore', category=(FutureWarning, RuntimeWarning))

class CellWithNetworkx:
    def __init__(self, swc_file, bg_exc_freq, bg_inh_freq, SIMU_DURATION, STIM_DURATION, 
                 syn_pos_seed, bg_spike_gen_seed, clus_spike_gen_seed=None, with_ap=False, with_global_rec=False):
        """
        Initialize cell with networkx structure.
        
        Args:
            swc_file: Path to SWC morphology file
            bg_exc_freq: Background excitatory frequency (Hz)
            bg_inh_freq: Background inhibitory frequency (Hz)
            SIMU_DURATION: Simulation duration (ms)
            STIM_DURATION: Stimulation duration (ms)
            syn_pos_seed: Random seed for synapse positioning (controls synapse locations, 
                            cluster positions, and synapse weights). Should be fixed across 
                            simulations to maintain consistent morphology.
            bg_spike_gen_seed: Random seed for background (bg) spike generation (bg spike trains, pink noise). Used by add_background_*_inputs.
            clus_spike_gen_seed: Random seed for cluster stimulus spike generation (stim times in
                          generate_vecstim, preunit permutation). If None, falls back to bg_spike_gen_seed.
            with_ap: If True, use L5PCbiophys3withNaCa.hoc (with AP and Ca), 
                    else use L5PCbiophys3.hoc (default: False)
            with_global_rec: If True, record seg_ina and seg_inmda for all segments and save to npy (default: False)
        """
        h.load_file("import3d.hoc")

        # h.nrn_load_dll('./mod/nrnmech.dll') # For Windows
        h.nrn_load_dll('./mod/x86_64/.libs/libnrnmech.so') # For Linux/Mac
        
        # Select biophysics file based on with_ap parameter
        biophys_file = './modelFile/L5PCbiophys3withNaCa.hoc' if with_ap else './modelFile/L5PCbiophys3.hoc'
        h.load_file(biophys_file)
        h.load_file('./modelFile/L5PCtemplate.hoc')

        self.complex_cell = h.L5PCtemplate(swc_file)
        h.celsius = 37
        
        h.v_init = self.complex_cell.soma[0].e_pas  # -90 mV

        self.distance_matrix = None

        self.num_syn_basal_exc = 0
        self.num_syn_apic_exc = 0
        self.num_syn_basal_inh = 0
        self.num_syn_apic_inh = 0
        self.num_syn_soma_inh = 0

        # Random seed for background (bg) spike generation (temporal dynamics)
        # Controls: bg spike trains, pink noise generation, firing patterns (add_background_*_inputs)
        self.bg_spike_gen_seed = bg_spike_gen_seed
        # Random seed for cluster stimulus: stim times in generate_vecstim, preunit permutation
        self.clus_spike_gen_seed = clus_spike_gen_seed if clus_spike_gen_seed is not None else bg_spike_gen_seed
        
        # Random seed for synapse positioning (spatial structure)
        # Controls: synapse locations, cluster positions, synapse weights
        self.syn_pos_seed = syn_pos_seed
        self.rnd = np.random.default_rng(syn_pos_seed)  # For synapse position selection
        random.seed(syn_pos_seed)  # For Python random.choices in add_single_synapse

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
        self.all_segments = [seg for sec in h.allsec() for seg in sec] 
        self.all_segments_noaxon = [seg for sec in self.all_sections for seg in sec]
                                       
        self.section_synapse_df = pd.DataFrame(columns=[
            'section_id_synapse', 'section_synapse', 'segment_synapse', 'loc', 'type',
            'distance_to_soma', 'distance_to_tuft', 'cluster_flag', 'cluster_center_flag',
            'cluster_id', 'pre_unit_id', 'region', 'branch_idx', 'syn_w', 'synapse',
            'netstim', 'netcon', 'spike_train', 'spike_train_bg'
        ], dtype=object)
                                         
        # For clustered synapses (will be assigned in assign_clustered_synapses)
        self.basal_channel_type = None
        self.sec_type = None
        self.num_clusters = None
        self.num_clusters_sampled = None
        self.cluster_radius = None

        # Input parameters (will be assigned in add_inputs)
        self.input_ratio_basal_apic = None
        self.bg_exc_channel_type = None
        self.initW = None
        self.num_func_group = None
        self.inh_delay = None

        # Stimulation parameters (will be assigned in assign_clustered_synapses)
        self.num_stim = None
        self.stim_time = None
        self.num_conn_per_preunit = None
        self.num_preunit = None

    

        # Cluster assignment (will be assigned in assign_clustered_synapses)
        self.unit_ids = None
        self.indices = None

        # Lists (will be assigned in add_inputs)
        self.num_syn_inh_list = None
        self.num_activated_preunit_list = None

        # Arrays (will be initialized in add_inputs)
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

        self.with_global_rec = with_global_rec
        self.seg_v_array = None
        self.seg_ina_array = None
        self.seg_inmda_array = None 

        self.lock = threading.Lock()

        self.section_df = pd.DataFrame(columns=[
            'parent_id', 'section_id', 'parent_name', 'section_name', 
            'length', 'branch_idx', 'section_type'
        ])
        
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
                           
    def add_single_synapse(self, num_syn, region, sim_type):
        
        type = 'A' if sim_type == 'exc' else 'B'
        
        region_mapping = {
            'basal': (self.sections_basal, 'dend'),
            'apical': (self.sections_apical, 'apic'),
            'soma': (self.sections_soma, 'soma')
        }
        sections, section_type = region_mapping[region]
        section_length = np.array(self.section_df.loc[self.section_df['section_type'] == section_type, 'length'])

        def generate_synapse(_):
            section = random.choices(sections, weights=section_length)[0][0].sec # rnd does not have a choices method
            section_name = section.psection()['name']
            
            section_id_synapse = self.section_df.loc[self.section_df['section_name'] == section_name, 'section_id'].iat[0]
            # self.section_df[self.section_df['section_name'] == section_name]['section_id'].values[0]
            branch_idx = self.section_df.loc[self.section_df['section_name'] == section_name, 'branch_idx'].iat[0]
            # self.section_df[self.section_df['section_name'] == section_name]['branch_idx'].values[0]   

            loc = self.rnd.uniform()
            segment_synapse = section(loc)
            
            distance_to_soma = recur_dist_to_soma(section, loc)
            distance_to_tuft = recur_dist_to_root(section, loc, self.root_tuft_sec) if section_id_synapse in self.sec_tuft_idx else -1 

            data_to_append = {
                'section_id_synapse': section_id_synapse, 'section_synapse': section, 'segment_synapse': segment_synapse,
                'loc': loc, 'type': type, 'distance_to_soma': distance_to_soma, 'distance_to_tuft': distance_to_tuft,
                'cluster_flag': -1, 'cluster_center_flag': -1, 'cluster_id': -1, 'pre_unit_id': -1,
                'region': region, 'branch_idx': branch_idx, 'syn_w': None, 'synapse': None,
                'netstim': None, 'netcon': None, 'spike_train': [], 'spike_train_bg': []
            }

            with self.lock:
                self.section_synapse_df = pd.concat([self.section_synapse_df, pd.DataFrame([data_to_append], dtype=object)], ignore_index=True)

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            list(tqdm(executor.map(generate_synapse, range(num_syn)), total=num_syn))

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

        num_syn_thres = [3000 + i * 3000 for i in range(2)] if sec_type == 'basal' else [2500 + i * 2500 for i in range(2)]

        # Get the indices for the thresholds
        dist_thres_basal = [0] + [sorted_basal_distances[threshold - 1] for threshold in num_syn_thres 
                                  if threshold <= len(sorted_basal_distances)] + [max(sorted_basal_distances)]

        dist_thres_tuft = [0] + [sorted_tuft_distances[threshold - 1] for threshold in num_syn_thres 
                                 if threshold <= len(sorted_tuft_distances)] + [max(sorted_tuft_distances)]

        # 
        num_conn_per_preunit = min(num_conn_per_preunit, num_clusters) 
        num_preunit = num_syn_per_clus * np.ceil(num_clusters / 3).astype(int)

        if spat_condition == 'clus':            
            # Number of synapses in each cluster is not fixed
            indices = generate_indices(self.rnd, num_clusters, num_conn_per_preunit, num_preunit)
            
            self.num_clusters_sampled = num_clusters

        elif spat_condition == 'distr':
            # num_pre*num_conn clus with 1 syn per 'cluster'
            num_clusters = num_preunit * num_conn_per_preunit
            numbers = np.repeat(np.arange(num_preunit), num_conn_per_preunit)
            self.rnd.shuffle(numbers)
            indices = [[num] for num in numbers]
            self.num_clusters_sampled = min(10, num_clusters)

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

        # Use syn_pos_seed for cluster positioning (spatial structure)
        clus_loc_rnd = np.random.RandomState(self.syn_pos_seed)
        
        for i in range(self.num_clusters):

            loop_count = 0
            # clus_loc_rnd = np.random.RandomState(self.syn_pos_seed + i)

            # Unassigned background synapses for surround synapses
            sec_syn_bg_exc_df = self.section_synapse_df[(self.section_synapse_df['type'] == 'A') & 
                                                        (self.section_synapse_df['cluster_flag'] == -1)]
                
            # Unassigned background synapses for center synapses
            # Define the concentration level for clus (6 clus on 6 branches / 1 branch)

            basal_branch_idx_list = [40, 41, 41]
            apic_branch_idx_list = [138, 138, 138]

            # Build DataFrame filter: common conditions + sec_type-specific conditions
            bg_exc_cond = (self.section_synapse_df['type'] == 'A') & (self.section_synapse_df['cluster_flag'] == -1)
            sec_specific_cond = (
                (self.section_synapse_df['region'] == 'basal') & 
                (self.section_synapse_df['distance_to_soma'].between(dist_thres_basal[dis_to_root], dist_thres_basal[dis_to_root+1]))
            ) if sec_type == 'basal' else (
                (self.section_synapse_df['section_id_synapse'].isin(self.sec_tuft_idx)) & 
                (self.section_synapse_df['distance_to_tuft'].between(dist_thres_tuft[dis_to_root], dist_thres_tuft[dis_to_root+1]))
            )
            sec_syn_bg_exc_ordered_df = self.section_synapse_df[bg_exc_cond & sec_specific_cond] 
                
            index_list = indices[i]
            num_syn_per_clus = len(index_list)  
            
            # Loop for cluster assignment
            while True:
                loop_count += 1

                # use the clus_loc_rnd for positioning
                syn_ctr = sec_syn_bg_exc_ordered_df.iloc[clus_loc_rnd.choice(len(sec_syn_bg_exc_ordered_df))]
                # syn_ctr = sec_syn_bg_exc_ordered_df.loc[clus_loc_rnd.choice(sec_syn_bg_exc_ordered_df.index)]
                print('syn_ctr:', syn_ctr['segment_synapse'])
                print('clus_branch_id:', syn_ctr['section_id_synapse'])
                
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
                        # dis_mark_from_ctr = np.sort(clus_loc_rnd.exponential(cluster_radius, num_syn_per_clus - 1))
                        max_dis_mark_from_ctr = np.sort(clus_loc_rnd.exponential(cluster_radius, max_num_syn_per_clus - 1))
                    except ValueError:
                        # dis_mark_from_ctr = np.sort(clus_loc_rnd.exponential(cluster_radius, 0))
                        max_dis_mark_from_ctr = np.sort(clus_loc_rnd.exponential(cluster_radius, 0))

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
                            syn_suc_sec_id = clus_loc_rnd.choice(list(self.DiG.successors(syn_suc_sec_id)))
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
                            syn_pre_sec_id = clus_loc_rnd.choice(list(self.DiG.predecessors(syn_pre_sec_id)))
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
                    clus_mem_idx = clus_loc_rnd.choice(max_clus_mem_idx, num_syn_per_clus - 1, replace=False)
                    # print('clus_mem_idx ver 1:', clus_mem_idx)
                    
                    # if self.num_preunit < 72:
                    #     clus_mem_max_size_idx = clus_loc_rnd.choice(max_clus_mem_idx, 72 - 1, replace=False)
                    #     perm = np.random.permutation(num_syn_per_clus - 1)
                    #     clus_mem_idx = clus_mem_max_size_idx[perm[:num_syn_per_clus - 1]]
                    #     print('clus_mem_idx ver 2:', clus_mem_idx)

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

        # Determine spat_condition and num_clus_condition based on folder_path
        if 'distr' in folder_path:
            spat_condition, num_clus_condition = 'distr', 'multi' if 'multiclus' in folder_path else 'single'
            section_synapse_df_clus = pd.read_csv(os.path.join(folder_path.replace('distr', 'clus'), 'section_synapse_df.csv'))
        elif 'multiclus' in folder_path:
            spat_condition, num_clus_condition = 'clus', 'multi'
            folder_path_clus = folder_path.replace('multiclus_3', 'singclus').replace('/G/results/simulation_multiclus_Oct25/', '/mnt/mimo_1/simu_results_sjc/simulation_singclus_Aug25/')
            parts = list(Path(folder_path_clus).parts)
            parts[-2] = '1'
            section_synapse_df_clus = pd.read_csv(os.path.join(Path(*parts).as_posix(), 'section_synapse_df.csv'))
        else:
            spat_condition, num_clus_condition, section_synapse_df_clus = 'clus', 'single', self.section_synapse_df

        # Cluster stimulus: use clus_spike_gen_seed for stim time generation and preunit order
        clus_spk_rnd = np.random.RandomState(self.clus_spike_gen_seed)

        spt_unit_array_list = []
        stim_time_var = 5
        for num_stim in range(1, self.num_stim + 1):
            spt_unit_array = generate_vecstim(clus_spk_rnd, self.unit_ids, num_stim, self.stim_time, stim_time_var)
            spt_unit_array_list.append(spt_unit_array)
        
        perm = clus_spk_rnd.permutation(self.num_preunit)
        
        ## Rearrange the perm to always start with the first syn of the first cluster
        pre_unit_id_first_syn = self.section_synapse_df[(self.section_synapse_df['cluster_center_flag'] == 1) &
                                                                (self.section_synapse_df['cluster_id'] == 0)]['pre_unit_id'].values[0]
        perm_list = perm.tolist()
        if pre_unit_id_first_syn in perm_list:
            perm_list.remove(pre_unit_id_first_syn)
            perm_list = [pre_unit_id_first_syn] + perm_list
        perm = np.array(perm_list)
                                                    
        # Format spt_unit_array with integer values for display
        spt_array_formatted = [(unit_id, arr.astype(int)) for unit_id, arr in spt_unit_array_list[0]]
        print('spt_unit_array:', spt_array_formatted)
        print('perm:', perm)
        print('indices', self.indices)

        self.num_syn_inh_list = [self.num_syn_basal_inh, self.num_syn_apic_inh, self.num_syn_soma_inh]
        
        # create an ndarray to store the voltage of each cluster of each trial 
        num_time_points = 1 + 40 * self.SIMU_DURATION
        
        if 'expected' in folder_path:
            iter_step = 1
        else:
            iter_step = 4

        # Generate preunit list with "dense first, sparse later" pattern
        # First include all integers from 0 to step (dense), then include step, step*2, step*3, ..., up to num_preunit (sparse)
        dense_part = list(range(0, iter_step + 1))
        sparse_part = list(range(iter_step, self.num_preunit + 1, iter_step))  # for sing-clus (add 1 is to allow the last num_preunit to be included)
        
        # self.num_activated_preunit_list = sorted(list(set(dense_part + sparse_part)))
        self.num_activated_preunit_list = [0, self.num_preunit] #[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24] # [0, 1, 3, 6, 12, 24, 48, 72] # [0, 3, 6, 9, 12, 18, 24] #[self.num_preunit] # for multi-clus
        num_aff_fibers = len(self.num_activated_preunit_list)
        
        # Initialize arrays with common shape
        common_shape = (num_time_points, self.num_stim, num_aff_fibers, num_trials)
        dend_shape = (self.num_clusters_sampled, *common_shape)
        
        # Initialize arrays concisely using dictionary and setattr
        voltage_arrays = ['soma_v', 'apic_v', 'apic_ica', 'soma_i', 'trunk_v', 'basal_v', 'tuft_v']
        bg_current_arrays = ['basal_bg_i_nmda', 'basal_bg_i_ampa', 'tuft_bg_i_nmda', 'tuft_bg_i_ampa']
        dend_arrays = ['dend_v', 'dend_i', 'dend_nmda_i', 'dend_ampa_i', 'dend_nmda_g', 'dend_ampa_g']
        for arr_name in voltage_arrays + bg_current_arrays:
            setattr(self, f'{arr_name}_array', np.zeros(common_shape))
        for arr_name in dend_arrays:
            setattr(self, f'{arr_name}_array', np.zeros(dend_shape))

        if self.with_global_rec:
            num_segments_noaxon = len(self.all_segments_noaxon)
            seg_global_shape = (num_segments_noaxon, num_time_points, self.num_stim, num_aff_fibers, num_trials)
            self.seg_v_array = np.zeros(seg_global_shape)
            self.seg_ina_array = np.zeros(seg_global_shape)
            self.seg_inmda_array = np.zeros(seg_global_shape)

        if simu_condition == 'invivo':
            add_background_exc_inputs(self.section_synapse_df, self.syn_param_exc, self.SIMU_DURATION, self.FREQ_EXC, 
                                    self.input_ratio_basal_apic, self.bg_exc_channel_type, self.initW, self.num_func_group,
                                    self.syn_pos_seed, self.bg_spike_gen_seed, spat_condition, num_clus_condition, section_synapse_df_clus)
        
        for num_activated_preunit in self.num_activated_preunit_list:  

            # if condition_met:
            #     break  # End the whole loop if the condition has been met

            for num_stim in range(self.num_stim):
                for num_trial in range(num_trials): # 20

                    # if simu_condition == 'invivo':
                    
                    # spt_unit_list_list = []
                    # for num_stim_idx in range(1, self.num_stim + 1):
                    #     spt_unit_list = generate_vecstim(self.unit_ids, num_stim_idx, self.stim_time)
                    #     spt_unit_list_list.append(spt_unit_list)

                    spt_unit_array = spt_unit_array_list[num_stim]
                    spt_unit_array_truncated = spt_unit_array[perm[:num_activated_preunit]]
                    # spt_unit_list_truncated = spt_unit_list

                    if 'expected' in folder_path and num_activated_preunit > 0:
                        # For expected input
                        spt_unit_array_truncated = spt_unit_array[perm[:num_activated_preunit][-1]]
                        
                    add_clustered_inputs(self.section_synapse_df, self.num_clusters, self.basal_channel_type, 
                                         self.initW, spt_unit_array_truncated, self.syn_pos_seed, self.num_preunit)
                    
                # for num_trial in range(num_trials): # 20

                    # Add background inputs for in vivo-like condition
                    if simu_condition == 'invivo':
                        num_activated_preunit_idx = self.num_activated_preunit_list.index(num_activated_preunit)
                        add_background_inh_inputs(self.section_synapse_df, self.syn_param_inh, self.SIMU_DURATION, self.FREQ_INH,  
                                                self.inh_delay, self.bg_spike_gen_seed, spat_condition, num_clus_condition,
                                                section_synapse_df_clus, num_activated_preunit_idx)
                
                for num_trial in range(num_trials):
                    num_aff_idx = self.num_activated_preunit_list.index(num_activated_preunit)

                    self.run_simulation(num_stim, num_aff_idx, num_trial, folder_path)

        # Save arrays efficiently
        arrays_to_save = {
            'soma_v_array': self.soma_v_array, 'apic_v_array': self.apic_v_array, 'apic_ica_array': self.apic_ica_array,
            'soma_i_array': self.soma_i_array, 'trunk_v_array': self.trunk_v_array, 'basal_v_array': self.basal_v_array,
            'tuft_v_array': self.tuft_v_array, 'basal_bg_i_nmda_array': self.basal_bg_i_nmda_array,
            'basal_bg_i_ampa_array': self.basal_bg_i_ampa_array, 'tuft_bg_i_nmda_array': self.tuft_bg_i_nmda_array,
            'tuft_bg_i_ampa_array': self.tuft_bg_i_ampa_array, 'dend_v_array': self.dend_v_array,
            'dend_i_array': self.dend_i_array, 'dend_nmda_i_array': self.dend_nmda_i_array,
            'dend_ampa_i_array': self.dend_ampa_i_array, 'dend_nmda_g_array': self.dend_nmda_g_array,
            'dend_ampa_g_array': self.dend_ampa_g_array
        }
        if self.with_global_rec:
            if self.seg_v_array is not None:
                arrays_to_save['seg_v_array'] = self.seg_v_array
            if self.seg_ina_array is not None:
                arrays_to_save['seg_ina_array'] = self.seg_ina_array
            if self.seg_inmda_array is not None:
                arrays_to_save['seg_inmda_array'] = self.seg_inmda_array

        for name, array in arrays_to_save.items():
            np.save(os.path.join(folder_path, f'{name}.npy'), array)

        self.section_synapse_df.to_csv(os.path.join(folder_path, 'section_synapse_df.csv'), index=False)
        # visualize_synapses(self.section_synapse_df, '/G/results/visualization_simulation_singclus')
        
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
            
            # choose the center synapse of each cluster (spatial condition: clus)
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

        
        # Repertoire of seg_v (voltage), seg_ina (Na current, built-in) and per-segment iNMDA; only when global recording is enabled
        seg_v = None
        seg_ina = None
        seg_inmda_vectors = None
        if self.with_global_rec:
            seg_v = [h.Vector().record(seg._ref_v) for seg in self.all_segments_noaxon]
            seg_ina = [h.Vector().record(seg._ref_ina) for seg in self.all_segments_noaxon]
            seg_to_syns = defaultdict(list)
            for _, row in self.section_synapse_df[self.section_synapse_df['type'] == 'A'].iterrows():
                seg_syn = row['segment_synapse']
                syn = row['synapse']
                if syn is not None and seg_syn is not None:
                    sec = seg_syn.sec
                    nseg = sec.nseg
                    seg_idx = min(int(seg_syn.x * nseg), nseg - 1)
                    seg_to_syns[(id(sec), seg_idx)].append(syn)
            seg_inmda_vectors = []
            for seg in self.all_segments_noaxon:
                seg_idx = min(int(seg.x * seg.sec.nseg), seg.sec.nseg - 1)
                key = (id(seg.sec), seg_idx)
                syns_on_seg = seg_to_syns.get(key, [])
                vecs = []
                for exc_syn in syns_on_seg:
                    try:
                        vecs.append(h.Vector().record(exc_syn._ref_i_NMDA))
                    except AttributeError:
                        vecs.append(h.Vector().record(exc_syn._ref_i_AMPA))
                seg_inmda_vectors.append(vecs)

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

        seg_inmda = None
        if self.with_global_rec and seg_inmda_vectors is not None:
            n_t = int(soma_v.size())
            seg_inmda = []
            for vecs in seg_inmda_vectors:
                if vecs:
                    seg_inmda.append(np.sum([np.array(v) for v in vecs], axis=0))
                else:
                    seg_inmda.append(np.zeros(n_t))
        
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

        # visualize_morpho(self.section_synapse_df, soma_v, seg_v, folder_path)

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
                self.dend_i_array[cluster_id, :, num_stim, num_aff_fiber, num_trial] = np.sum(np.array(dend_i_list_list[cluster_id]), axis=0) # sum, not average
                self.dend_nmda_i_array[cluster_id, :, num_stim, num_aff_fiber, num_trial] = np.sum(np.array(dend_i_nmda_list_list[cluster_id]), axis=0)
                self.dend_nmda_g_array[cluster_id, :, num_stim, num_aff_fiber, num_trial] = np.sum(np.array(dend_g_nmda_list_list[cluster_id]), axis=0)
                
                self.dend_ampa_i_array[cluster_id, :, num_stim, num_aff_fiber, num_trial] = np.sum(np.array(dend_i_ampa_list_list[cluster_id]), axis=0)
                self.dend_ampa_g_array[cluster_id, :, num_stim, num_aff_fiber, num_trial] = np.sum(np.array(dend_g_ampa_list_list[cluster_id]), axis=0)

            if self.with_global_rec and seg_v is not None and seg_ina is not None and seg_inmda is not None:
                self.seg_v_array[:, :, num_stim, num_aff_fiber, num_trial] = np.array([list(v) for v in seg_v])
                self.seg_ina_array[:, :, num_stim, num_aff_fiber, num_trial] = np.array([list(v) for v in seg_ina])
                self.seg_inmda_array[:, :, num_stim, num_aff_fiber, num_trial] = np.array(seg_inmda)

        return True
      
# main function
swc_file_path = './modelFile/cell1.asc'

def create_parser():
    """Create and configure argument parser with default values from utils"""
    parser = argparse.ArgumentParser(description='Neuron simulation parameters')
    
    # Synapse numbers
    parser.add_argument('--num_syn_basal_exc', type=int, default=10042,
                        help='Number of basal excitatory synapses (default: 10042)')
    parser.add_argument('--num_syn_apic_exc', type=int, default=16070,
                        help='Number of apical excitatory synapses (default: 16070)')
    parser.add_argument('--num_syn_basal_inh', type=int, default=1023,
                        help='Number of basal inhibitory synapses (default: 1023)')
    parser.add_argument('--num_syn_apic_inh', type=int, default=1637,
                        help='Number of apical inhibitory synapses (default: 1637)')
    parser.add_argument('--num_syn_soma_inh', type=int, default=150,
                        help='Number of soma inhibitory synapses (default: 150)')
    
    # Simulation duration
    parser.add_argument('--simu_duration', type=int, default=1000,
                        help='Simulation duration in ms (default: 1000)')
    parser.add_argument('--stim_duration', type=int, default=1000,
                        help='Stimulation duration in ms (default: 1000)')
    parser.add_argument('--stim_time', type=int, default=500,
                        help='Time point of stimulation in ms (default: 500)')
    parser.add_argument('--num_stim', type=int, default=1,
                        help='Number of stimuli (default: 1)')
    
    # Channel types
    parser.add_argument('--basal_channel_type', type=str, default='AMPANMDA',
                        choices=['AMPANMDA', 'AMPA'],
                        help='Basal channel type (default: AMPANMDA)')
    parser.add_argument('--bg_exc_channel_type', type=str, default='AMPANMDA',
                        choices=['AMPANMDA', 'AMPA'],
                        help='Background excitatory channel type (default: AMPANMDA)')
    parser.add_argument('--channel_suffix', type=str, default='singclus',
                        help='Base channel suffix for simulation folder name (e.g. singclus, singclus_AMPA). '
                             'With --with_ap / --with_global_rec, _ap / _globrec are appended (default: singclus)')
    
    # Cluster parameters
    parser.add_argument('--cluster_radius', type=float, default=5.0,
                        help='Cluster radius in um (default: 5.0)')
    parser.add_argument('--num_clusters', type=int, default=1,
                        help='Number of clusters (default: 1)')
    parser.add_argument('--num_syn_per_clus', type=int, default=72,
                        help='Number of synapses per cluster (default: 72)')
    parser.add_argument('--num_conn_per_preunit', type=int, default=3,
                        help='Number of connections per preunit (default: 3)')
    
    # Simulation conditions
    parser.add_argument('--simu_condition', type=str, default='invivo',
                        choices=['invivo', 'invitro'],
                        help='Simulation condition (default: invivo)')
    parser.add_argument('--spat_condition', type=str, default='clus',
                        choices=['clus', 'distr'],
                        help='Spatial condition: clus (clustered) or distr (distributed) (default: clus)')
    parser.add_argument('--sec_type', type=str, default='basal',
                        choices=['basal', 'apical'],
                        help='Section type (default: basal)')
    parser.add_argument('--distance_to_root', type=int, default=0,
                        help='Distance from clusters to root (default: 0)')
    
    # Background input parameters
    parser.add_argument('--bg_exc_freq', type=float, default=1.0,
                        help='Background excitatory frequency in Hz (default: 1.0)')
    parser.add_argument('--bg_inh_freq', type=float, default=4.0,
                        help='Background inhibitory frequency in Hz (default: 4.0)')
    parser.add_argument('--input_ratio_basal_apic', type=float, default=1.0,
                        help='Input ratio of basal to apical (default: 1.0)')
    
    # Synaptic weight parameters
    parser.add_argument('--initW', type=float, default=0.0004,
                        help='Initial weight of AMPANMDA synapses in uS (default: 0.0004)')
    parser.add_argument('--num_func_group', type=int, default=10,
                        help='Number of functional groups (default: 10)')
    parser.add_argument('--inh_delay', type=float, default=4.0,
                        help='Delay of inhibitory inputs in ms (default: 4.0)')
    
    # Other parameters
    parser.add_argument('--num_trials', type=int, default=1,
                        help='Number of trials (default: 1)')
    parser.add_argument('--folder_tag', type=str, default='1',
                        help='Folder tag for output (default: 1)')
    # parser.add_argument('--epoch', type=int, default=1,
    #                     help='Epoch number (default: 1)')
    
    # Random seeds - both default to epoch value
    parser.add_argument('--syn_pos_seed', type=int, default=None,
                        help='Random seed for synapse positioning (locations, clusters, weights). '
                             'If None, uses epoch value. Controls spatial structure (default: None)')
    parser.add_argument('--bg_spike_gen_seed', type=int, default=None,
                        help='Random seed for background (bg) spike generation (bg spike trains, pink noise). If None, uses epoch value (default: None)')
    parser.add_argument('--clus_spike_gen_seed', type=int, default=None,
                        help='Random seed for cluster stimulus spike generation (stim times in generate_vecstim, '
                             'preunit permutation). If None, uses epoch value. Distinct from bg_spike_gen_seed (bg only) (default: None)')
    
    # Biophysics model selection
    # Default is False (use L5PCbiophys3.hoc without AP and Ca)
    # Use --with_ap to enable L5PCbiophys3withNaCa.hoc (with AP and Ca dynamics)
    parser.add_argument('--with_ap', action='store_true', default=False,
                        help='Use L5PCbiophys3withNaCa.hoc (with AP and Ca dynamics). '
                             'Default is False (uses L5PCbiophys3.hoc without AP and Ca)')
    # Global segment recording: seg_ina and seg_inmda for all segments, saved as npy
    parser.add_argument('--with_global_rec', action='store_true', default=False,
                        help='Record seg_ina and seg_inmda for all segments and save to npy. Default is False')
    
    return parser

def build_cell(args):
    """Build and simulate cell with parameters from argparse"""
    
    # Extract parameters from args using getattr (more concise than individual assignments)
    def get_param(name): return getattr(args, name)
    
    # Get epoch first for separator, then extract all parameters
    epoch = get_param('epoch')
    print('\n' + '='*80)
    print(f'EPOCH {epoch}')
    print('='*80 + '\n')
    
    # Core parameters
    NUM_SYN_BASAL_EXC, NUM_SYN_APIC_EXC = get_param('num_syn_basal_exc'), get_param('num_syn_apic_exc')
    NUM_SYN_BASAL_INH, NUM_SYN_APIC_INH = get_param('num_syn_basal_inh'), get_param('num_syn_apic_inh')
    NUM_SYN_SOMA_INH, SIMU_DURATION, STIM_DURATION = get_param('num_syn_soma_inh'), get_param('simu_duration'), get_param('stim_duration')
    simu_condition, spat_condtion = get_param('simu_condition'), get_param('spat_condition')
    basal_channel_type, sec_type = get_param('basal_channel_type'), get_param('sec_type')
    distance_to_root, num_clusters, cluster_radius = get_param('distance_to_root'), get_param('num_clusters'), get_param('cluster_radius')
    bg_exc_freq, bg_inh_freq = get_param('bg_exc_freq'), get_param('bg_inh_freq')
    input_ratio_basal_apic, bg_exc_channel_type = get_param('input_ratio_basal_apic'), get_param('bg_exc_channel_type')
    initW, num_func_group, inh_delay = get_param('initW'), get_param('num_func_group'), get_param('inh_delay')
    num_stim, stim_time = get_param('num_stim'), get_param('stim_time')
    num_conn_per_preunit, num_syn_per_clus = get_param('num_conn_per_preunit'), get_param('num_syn_per_clus')
    num_trials, folder_tag = get_param('num_trials'), get_param('folder_tag')
    
    # Random seeds: default to epoch if not set
    # syn_pos_seed: spatial structure (synapse positions, clusters, weights)
    # bg_spike_gen_seed: background (bg) spike generation (bg spike trains, pink noise)
    # clus_spike_gen_seed: cluster stimulus (stim times in generate_vecstim, preunit permutation)
    syn_pos_seed = args.syn_pos_seed if args.syn_pos_seed is not None else epoch
    bg_spike_gen_seed = args.bg_spike_gen_seed if args.bg_spike_gen_seed is not None else epoch
    clus_spike_gen_seed = args.clus_spike_gen_seed if args.clus_spike_gen_seed is not None else epoch
    with_ap, with_global_rec = args.with_ap, args.with_global_rec
  
    # time_tag = time.strftime("%Y%m%d", time.localtime())
    # folder_path = '/G/results/simulation/' + time_tag + '/' + folder_tag          
    # Build channel_suffix: ensure leading underscore, then append conditional suffixes
    channel_suffix = args.channel_suffix.strip()
    channel_suffix = ('_' + channel_suffix) if channel_suffix and not channel_suffix.startswith('_') else channel_suffix
    channel_suffix += ''.join(['_ap' if with_ap else '', '_globrec' if with_global_rec else ''])
    simu_folder = f'{sec_type}_range{distance_to_root}_{spat_condtion}_{simu_condition}{channel_suffix}'
    
    # Normalize folder tag
    folder_tag = str(int(folder_tag) % 100) if int(folder_tag) % 100 != 0 else '100'
    folder_path = f'/G/results/simulation_singclus_supple_Feb26/{simu_folder}/{folder_tag}/{epoch}'
    # folder_path = Path('/G/results/simulation_multiclus_Oct25') / simu_folder / folder_tag / str(epoch)

    simulation_params = {
        'cell model': 'L5PN',
        'NUM_SYN_BASAL_EXC': NUM_SYN_BASAL_EXC, 'NUM_SYN_APIC_EXC': NUM_SYN_APIC_EXC,
        'NUM_SYN_BASAL_INH': NUM_SYN_BASAL_INH, 'NUM_SYN_APIC_INH': NUM_SYN_APIC_INH,
        'NUM_SYN_SOMA_INH': NUM_SYN_SOMA_INH, 'SIMU DURATION': SIMU_DURATION,
        'STIM DURATION': STIM_DURATION, 'simulation condition': simu_condition,
        'synaptic spatial condition': spat_condtion, 'basal channel type': basal_channel_type,
        'channel_suffix': args.channel_suffix, 'section type': sec_type,
        'distance from clusters to root': distance_to_root, 'number of clusters': num_clusters,
        'cluster radius': cluster_radius, 'background excitatory frequency': bg_exc_freq,
        'background inhibitory frequency': bg_inh_freq, 'input ratio of basal to apical': input_ratio_basal_apic,
        'background excitatory channel type': bg_exc_channel_type, 'initial weight of AMPANMDA synapses': initW,
        'number of functional groups': num_func_group, 'delay of inhibitory inputs': inh_delay,
        'number of stimuli': num_stim, 'time point of stimulation': stim_time,
        'number of connection per preunit': num_conn_per_preunit, 'number of synapses per cluster': num_syn_per_clus,
        'number of trials': num_trials, 'syn_pos_seed': syn_pos_seed,
        'bg_spike_gen_seed': bg_spike_gen_seed, 'clus_spike_gen_seed': clus_spike_gen_seed,
        'with_ap': with_ap, 'with_global_rec': with_global_rec,
    }

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    json_filename = os.path.join(folder_path, "simulation_params.json")
    with open(json_filename, 'w') as json_file:
        json.dump(simulation_params, json_file, indent=4)

    cell1 = CellWithNetworkx(swc_file_path, bg_exc_freq, bg_inh_freq, SIMU_DURATION, STIM_DURATION, 
                            syn_pos_seed, bg_spike_gen_seed, clus_spike_gen_seed, with_ap, with_global_rec)
    cell1.add_synapses(NUM_SYN_BASAL_EXC, NUM_SYN_APIC_EXC, NUM_SYN_BASAL_INH, NUM_SYN_APIC_INH, NUM_SYN_SOMA_INH)
    
    cell1.assign_clustered_synapses(basal_channel_type, sec_type, distance_to_root, 
                                    num_clusters, cluster_radius, num_stim, stim_time, 
                                    spat_condtion, num_conn_per_preunit, num_syn_per_clus, folder_path) 

    cell1.add_inputs(folder_path, simu_condition, input_ratio_basal_apic, 
                     bg_exc_channel_type, initW, num_func_group, inh_delay, num_trials)

def run_processes(args_list, epoch):
    """Run multiple processes with different parameter sets"""
    processes = []  # Create a new process list for each set of parameters
    for args in args_list:
        # Create a copy of args and set epoch
        args_copy = argparse.Namespace(**vars(args))
        args_copy.epoch = epoch
        process = multiprocessing.Process(target=build_cell, args=(args_copy,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()  # Join each batch of processes before moving to the next parameter set

def run_combination(combination_args):
    """
    Run combination of parameters.
    
    Note: spat_cond (spatial condition) is processed sequentially within this function,
    not in parallel. This ensures all 'clus' tasks complete before any 'distr' tasks start.
    This sequential processing is intentional to maintain execution order.
    """
    sec_type, dis_to_root, epoch, base_args = combination_args
    
    # Process spatial conditions sequentially: first 'clus', then 'distr'
    # This ensures all clustered simulations complete before distributed ones begin
    for spat_cond in ['clus', 'distr']:
        # Create a copy of base_args (which contains command-line arguments)
        args = argparse.Namespace(**vars(base_args))
        # Override with combination-specific values
        args.sec_type = sec_type
        args.distance_to_root = dis_to_root
        args.spat_condition = spat_cond
        
        run_processes([args], epoch)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()  # Parse command-line arguments once
    
    # Running for sing-cluster analysis (nonlinearity)
    # Parameter combinations configuration - easy to modify and maintain
    param_config = {
        'sec_type': ['basal', 'apical'],           # Section types: ['basal', 'apical']
        'dis_to_root': [1],              # Distance to root: [0, 1, 2]
        # 'spat_cond': ['clus', 'distr'],           # Spatial condition: ['clus', 'distr']
        'batch_config': {
            'num_batches': 10,            # Number of batches
            'epochs_per_batch': 10,       # Epochs per batch
            'start_epoch': 1             # Starting epoch number
        }
    }
    
    # Generate all parameter combinations using itertools.product
    batch_config = param_config['batch_config']
    for batch_idx in range(batch_config['num_batches']):
        start_epoch = batch_config['start_epoch'] + batch_idx * batch_config['epochs_per_batch']
        end_epoch = start_epoch + batch_config['epochs_per_batch']
        
        # Generate all combinations of parameters
        # Note: spat_cond is NOT included here - it will be processed sequentially 
        # inside run_combination() to ensure 'clus' completes before 'distr' starts
        combinations = [
            (sec_type, dis_to_root, epoch, args)
            for sec_type, dis_to_root in itertools.product(
                param_config['sec_type'],
                param_config['dis_to_root']
            )
            for epoch in range(start_epoch, end_epoch)
        ]
        
        # Execute combinations in parallel
        # Each combination will internally process 'clus' then 'distr' sequentially
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            executor.map(run_combination, combinations)


    # multiprocessing.set_start_method('spawn', force=True) # Use spawn will initiate too many NEURON instances 

    # # Running for multi-cluster analysis
    # parser = create_parser()
    # for sec_type in ['basal']: # ['basal', 'apical']
    #     for spat_cond in ['clus', 'distr']: # ['clus', 'distr']
    #         for dis_to_root in [0, 2]: # [0, 1, 2]
    #             args_list = []
    #             base_args = parser.parse_args([])
    #             base_args.sec_type = sec_type
    #             base_args.spat_condition = spat_cond
    #             base_args.distance_to_root = dis_to_root
    #             args_list.append(base_args)
    #             for epoch in range(1, 6):
    #                 run_processes(args_list, epoch)

