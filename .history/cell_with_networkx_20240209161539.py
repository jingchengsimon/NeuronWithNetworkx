from neuron import h
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

from utils.visualize_utils import visualize_simulation, visualize_summary_simulation
from utils.graph_utils import create_graph, set_graph_order
from utils.add_inputs_utils import add_background_exc_inputs, add_background_inh_inputs, add_clustered_inputs
from utils.distance_utils import distance_synapse_mark_compare
from utils.create_stim_utils import generate_indices, get_stim_ids, create_vecstim
from utils.count_spikes import count_spikes

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=numba.NumbaDeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning) # remember update df.append to pd.concat

class CellWithNetworkx:
    def __init__(self, swc_file,bg_syn_freq):
        h.load_file("import3d.hoc")
        
        # changing the loc of folder seems to make loading the relative path wrong
        current_directory = os.path.dirname(__file__)  # 如果在脚本中使用，__file__是指当前脚本的文件名
        os.chdir(current_directory)
        relative_path = './mod/nrnmech.dll'
        nrnmech_path = os.path.join(current_directory, relative_path)
        h.nrn_load_dll(nrnmech_path)

        h.load_file('./modelFile/L5PCbiophys3.hoc')
        h.load_file('./modelFile/L5PCtemplate.hoc')
        self.complex_cell = h.L5PCtemplate(swc_file)
        h.celsius = 37
        h.v_init = self.complex_cell.soma[0].e_pas

        self.distance_matrix = None

        self.num_syn_basal_exc = 0
        self.num_syn_apic_exc = 0
        self.num_syn_basal_inh = 0
        self.num_syn_apic_inh = 0
        self.num_syn_clustered = 0

        # we should have 2 rnd, the one for positioning should be fixed through the simu
        # while the one for generating spikes should be different for each simu
        self.rnd = np.random.RandomState(10) 
        
        if bg_syn_freq != 0:
            self.spike_interval = 1000/bg_syn_freq # interval=1000(ms)/f
        self.time_interval = 1/1000 # 1ms, 0.001s
        self.FREQ_INH = 10  # Hz, /s
        self.DURATION = 1000

        self.syn_param_exc = [0, 0.3, 1.8, 0.0016] # reverse_potential, tau1, tau2, syn_weight
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
                                                'cluster_id',
                                                'pre_unit_id',
                                                'region']) # for adding vecstim of different orientation
                                        
    
        # self.spike_counts_basal_inh = None
        # self.spike_counts_apic_inh = None

        # For clustered synapses

        # self.num_syn_clustered = None
        self.num_clusters = None
        self.cluster_radius = None
        self.distance_to_soma = None
        self.num_conn_per_preunit = None
        self.num_syn_per_cluster = None
        self.basal_channel_type = None

        self.ori_dg_list = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        self.pref_ori_dg = None

        self.unit_ids = None
        self.indices = None

        self.soma_v_array = None
        self.dend_v_array = None
        self.dend_i_array = None
        self.dend_nmda_i_array = None
        self.dend_ampa_i_array = None

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
        
        # create section_df, G, DiG by graph_utils
        self.section_df, self.G, self.DiG, self.sp = create_graph(self.all_sections, self.section_df)
        # assign the order for each section
        self.class_dict = set_graph_order(self.G)

        # self._create_graph()
        # self._set_graph_order()

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

    def assign_clustered_synapses(self, num_clusters, cluster_radius, distance_to_soma, num_conn_per_preunit, num_syn_per_cluster, basal_channel_type):
        # self.pref_ori_dg, self.unit_ids, indices = generate_indices(self.rnd, num_clusters, num_conn_per_preunit)
        num_syn_per_cluster = num_syn_per_cluster
        indices = [[1] * num_syn_per_cluster]
        self.indices = indices
        
        self.num_clusters = num_clusters
        self.cluster_radius = cluster_radius
        self.distance_to_soma = distance_to_soma
        self.num_conn_per_preunit = num_conn_per_preunit
        self.num_syn_per_cluster = num_syn_per_cluster
        self.basal_channel_type = basal_channel_type

        # sec_syn_bg_exc_df = self.section_synapse_df[self.section_synapse_df['type'] == 'A']

        dist_list = self.class_dict.get(distance_to_soma, [])
        # sections directly connected to soma
        sections_k_distance = [section[0].sec for i, section in enumerate(self.all_sections) if i in dist_list]
        
        for i in range(num_clusters):
            sec_syn_bg_exc_df = self.section_synapse_df[(self.section_synapse_df['type'] == 'A')]

            sec_syn_bg_exc_ordered_df = self.section_synapse_df[
                (self.section_synapse_df['section_synapse'].isin(sections_k_distance)) & 
                (self.section_synapse_df['type'] == 'A') &
                (self.section_synapse_df['region'] == 'basal')]

            index_list = indices[0]
            # use the rnd for positioning
            num_syn_per_cluster = len(index_list)
            syn_ctr = sec_syn_bg_exc_ordered_df.loc[self.rnd.choice(sec_syn_bg_exc_ordered_df.index)]
            
            # assign the center as clustered synapse
            self.section_synapse_df.loc[syn_ctr.name, 'type'] = 'C'
            self.section_synapse_df.loc[syn_ctr.name, 'cluster_id'] = i
            self.section_synapse_df.loc[syn_ctr.name, 'pre_unit_id'] = index_list[0]

            syn_ctr_sec = syn_ctr['section_synapse']
            syn_surround_ctr = sec_syn_bg_exc_ordered_df[
                (sec_syn_bg_exc_ordered_df['section_synapse'] == syn_ctr_sec) & 
                (sec_syn_bg_exc_ordered_df.index != syn_ctr.name)]

            dis_syn_from_ctr = np.array(np.abs(syn_ctr['loc'] - syn_surround_ctr['loc']) * syn_ctr_sec.L)
            # use exponential distribution to generate loc
            dis_mark_from_ctr = np.sort(self.rnd.exponential(cluster_radius, num_syn_per_cluster - 1))

            # not enough synapses on the same section
            syn_ctr_sec_id = syn_ctr['section_id_synapse']
            syn_suc_sec_id = syn_ctr_sec_id
            syn_pre_sec_id = syn_ctr_sec_id
            while len(dis_syn_from_ctr) < num_syn_per_cluster - 1:
                # the children section of the center section
                if list(self.DiG.successors(syn_ctr_sec_id)):
                    # iterate
                    syn_suc_sec_id = self.rnd.choice(list(self.DiG.successors(syn_suc_sec_id)))
                    syn_suc_sec = sec_syn_bg_exc_df[sec_syn_bg_exc_df['section_id_synapse'] == syn_suc_sec_id]['section_synapse'].values[0]
                    syn_suc_surround_ctr = sec_syn_bg_exc_df[sec_syn_bg_exc_df['section_id_synapse'] == syn_suc_sec_id]
                    dis_syn_suc_from_ctr = np.array((1 - syn_ctr['loc']) * syn_ctr_sec.L + syn_suc_surround_ctr['loc'] * syn_suc_sec.L)
                    
                # the parent section of the center section
                # there is no section on the soma, so we should not choose soma as the parent section
                if list(self.DiG.predecessors(syn_ctr_sec_id)) not in ([], [0]):
                    syn_pre_sec_id = self.rnd.choice(list(self.DiG.predecessors(syn_pre_sec_id)))
                    syn_pre_sec = sec_syn_bg_exc_df[sec_syn_bg_exc_df['section_id_synapse'] == syn_pre_sec_id]['section_synapse'].values[0]
                    syn_pre_surround_ctr = sec_syn_bg_exc_df[sec_syn_bg_exc_df['section_id_synapse'] == syn_pre_sec_id]
                    dis_syn_pre_from_ctr = np.array(syn_ctr['loc'] * syn_ctr_sec.L + (1 - syn_pre_surround_ctr['loc']) * syn_pre_sec.L)

                dis_syn_from_ctr = np.concatenate((dis_syn_from_ctr, dis_syn_suc_from_ctr, dis_syn_pre_from_ctr))
                syn_surround_ctr = pd.concat([syn_surround_ctr, syn_suc_surround_ctr, syn_pre_surround_ctr])

            cluster_member_index = distance_synapse_mark_compare(dis_syn_from_ctr, dis_mark_from_ctr)
            # assign the surround as clustered synapse
            self.section_synapse_df.loc[syn_surround_ctr.iloc[cluster_member_index].index, 'type'] = 'C'
            self.section_synapse_df.loc[syn_surround_ctr.iloc[cluster_member_index].index, 'cluster_id'] = i
            self.section_synapse_df.loc[syn_surround_ctr.iloc[cluster_member_index].index, 'pre_unit_id'] = index_list[1:]

            print('cluster_id: ', i)
            print(self.section_synapse_df[self.section_synapse_df['cluster_id'] == i]['segment_synapse'].values)

    def add_inputs(self, folder_path):

        ori_dg_list, unit_ids, num_stims = self.ori_dg_list, self.unit_ids, 2
        # 创建一个空的 DataFrame
        self.num_spikes_df = pd.DataFrame(index=range(1, num_stims + 1), columns=ori_dg_list)
        # spt_unit_list = create_vecstim(ori_dg, stim_id, unit_ids)
        spt_unit_list = []
        
        add_background_exc_inputs(self.section_synapse_df, 
                                       self.syn_param_exc, 
                                       self.spike_interval, 
                                       self.lock)

        num_syn_inh = [self.num_syn_basal_inh, self.num_syn_apic_inh]
        # dend_v_list, dend_v_peak_list = [], []
        # soma_v_list, soma_v_peak_list = [], []
        
        for num_syn_to_get_input in range(self.num_syn_per_cluster):
            
            add_clustered_inputs(self.section_synapse_df, 
                                       self.syn_param_exc, 
                                       self.num_clusters, 
                                       self.num_conn_per_preunit, 
                                       self.basal_channel_type, 
                                       num_syn_to_get_input,
                                       spt_unit_list)
            
            add_background_inh_inputs(self.section_synapse_df, 
                                      self.syn_param_inh, 
                                      self.time_interval, 
                                      self.DURATION, 
                                      self.FREQ_INH, 
                                      num_syn_inh, 
                                      self.lock)
            
            ori_dg = stim_id = stim_index = 1
            # stim_index = np.where(stim_ids == stim_id)[0][0] + 1
            
            # create an ndarray to store the voltage of each cluster of each trial 
            num_time_points = 40000
            num_trials = 5

            self.dend_v_array = np.zeros((self.num_clusters, num_time_points, self.num_syn_per_cluster, num_trials))
            self.dend_i_array = np.zeros((self.num_clusters, num_time_points, self.num_syn_per_cluster, num_trials))
            self.dend_nmda_i_array = np.zeros((self.num_clusters, num_time_points, self.num_syn_per_cluster, num_trials))
            self.dend_ampa_i_array = np.zeros((self.num_clusters, num_time_points, self.num_syn_per_cluster, num_trials))
            self.soma_v_array = np.zeros((num_time_points, self.num_syn_per_cluster, num_trials))

            # For analysing the supra-linearity by NMDA
            # time_v, dend_v, soma_v = 
            self.run_simulation(ori_dg, stim_id, stim_index, num_syn_to_get_input, trial_idx, folder_path)
            
        #     dend_v_peak = np.max(np.array(dend_v)[20000:24000])
        #     dend_v_list.append(dend_v)
        #     dend_v_peak_list.append(dend_v_peak)

        #     soma_v_peak = np.max(np.array(soma_v)[20000:24000])
        #     soma_v_list.append(soma_v)
        #     soma_v_peak_list.append(soma_v_peak)

        # visualize_summary_simulation(time_v, dend_v_list, dend_v_peak_list, soma_v_list, soma_v_peak_list, folder_path)
            
        # for ori_dg in ori_dg_list: 
            # self._process_ori_dg(self, ori_dg, unit_ids, num_stims)
        
        # self.num_spikes_df.to_csv('./num_spikes_df_oarallel.csv', encoding='utf-8', index=False)
    
    def run_simulation(self, ori_dg, stim_id, stim_index, num_syn_to_get_input, trial_idx, folder_path):

        soma_v = h.Vector().record(self.complex_cell.soma[0](0.5)._ref_v)
        time_v = h.Vector().record(h._ref_t)
        apic_v = h.Vector().record(self.complex_cell.apic[0](0.5)._ref_v)

        # 创建用于保存所有 cluster 记录的列表
        dend_v_list = []
        dend_i_list = []
        dend_i_nmda_list = []
        dend_i_ampa_list = []

        # 假设 clusters 是包含所有 cluster 的列表或数组
        for cluster_id in range(self.num_clusters):
            # 获取当前 cluster 的 basal_ctr 和 syn_ctr
            cluster_basal_ctr = self.section_synapse_df[self.section_synapse_df['cluster_id'] == cluster_id]['segment_synapse'].values[0]
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

        # cluster_basal_ctr = self.section_synapse_df[self.section_synapse_df['cluster_id'] == 0]['segment_synapse'].values[0]
        # syn_ctr = self.section_synapse_df[self.section_synapse_df['cluster_id'] == 0]['synapse'].values[0]
        # dend_v = h.Vector().record(cluster_basal_ctr._ref_v)
        # dend_i = h.Vector().record(syn_ctr._ref_i)

        # try:
        #     dend_i_nmda = h.Vector().record(syn_ctr._ref_i_NMDA)
        # except AttributeError:
        #     dend_i_nmda = h.Vector().record(syn_ctr._ref_i_AMPA)
        
        # dend_i_ampa = h.Vector().record(syn_ctr._ref_i_AMPA)
    

        # netcons_list = self.section_synapse_df['netcon']
        # spike_times = [h.Vector() for _ in netcons_list]

        # clustered_syn_index = self.section_synapse_df[self.section_synapse_df['type'] == 'C'].index
        # spike_times_clustered = [spike_times[i] for i in clustered_syn_index]

        # for nc, spike_times_vec in zip(netcons_list, spike_times):
            # nc.record(spike_times_vec)

        h.tstop = self.DURATION
        st = time.time()
        h.run()
        print('complex cell simulation time {:.4f}'.format(time.time()-st))
              
        self.soma_v_array[:, trial_idx] = np.array(soma_v)
        for cluster_id in self.num_clusters:
            self.dend_v_array[cluster_id, :, num_syn_to_get_input, trial_idx] = np.array(dend_v_list[cluster_id])
        
        
        # # 累加多个spike trains得到总的放电次数
        # total_spikes = np.zeros(self.DURATION)  # 初始化总spike数数组

        # for spike_times_vec in spike_times:
        #     try:
        #         if np.all(np.floor(spike_times_vec).astype(int) < self.DURATION):
        #             total_spikes[np.floor(spike_times_vec).astype(int)] += 1 # 向下取整
        #     except ValueError:
        #         continue

        # # 计算每个时间点的平均firing rate (Hz, /s)
        # firing_rates = total_spikes / (np.mean(total_spikes) * self.time_interval)

        # threshold to define spikes
        threshold = 0

        num_spikes = count_spikes(soma_v, threshold)
        # print(str(ori_dg)+'-'+str(stim_id))
        print("Number of spikes:", num_spikes)
        
        self.num_spikes_df.at[stim_index, ori_dg] = num_spikes
        # self.num_spikes_list.append(num_spikes)

        # visualization_params = [ori_dg, stim_id, soma_v, dend_v, apic_v, time_v, spike_times_clustered, dend_i, dend_i_nmda, dend_i_ampa, spike_times]
        # visualize_simulation(visualization_params, num_syn_to_get_input, folder_path)

        # return time_v, dend_v, soma_v
    
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
            
            # synapse = h.Exp2Syn(segment_synapse)
            # synapse.e = e_syn
            # synapse.tau1 = tau1
            # synapse.tau2 = tau2

            data_to_append = {'section_id_synapse': section_id_synapse,
                            'section_synapse': section,
                            'segment_synapse': segment_synapse,
                            'synapse': None, 
                            'netstim': None,
                            'random': None,
                            'netcon': None,
                            'loc': loc,
                            'type': type,
                            'cluster_id': -1,
                            'pre_unit_id': -1,
                            'region': region}

            with self.lock:
                self.section_synapse_df = self.section_synapse_df.append(data_to_append, ignore_index=True)

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            list(tqdm(executor.map(generate_synapse, range(num_syn)), total=num_syn))
 
    def process_ori_dg(self, ori_dg, unit_ids, num_stims):

        # for ori_dg in ori_dg_list:
        stim_ids = get_stim_ids(ori_dg)
        for stim_id in stim_ids[:num_stims]:
            spt_unit_list = create_vecstim(ori_dg, stim_id, unit_ids)
            self._add_background_exc_inputs()
            self._add_clustered_inputs(spt_unit_list)
            self._add_background_inh_inputs()

            stim_index = np.where(stim_ids == stim_id)[0][0] + 1
            self._run_simulation(ori_dg, stim_id, stim_index)

    

