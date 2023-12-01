from neuron import gui, h
from neuron.units import ms, mV
import numpy as np
import time
import pandas as pd
import networkx as nx
from tqdm import tqdm
import warnings
import re
import random
from math import floor
import matplotlib.pyplot as plt
import seaborn as sns
import numba 
from numba import jit
from scipy.ndimage import gaussian_filter1d
import glob
import os
from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=numba.NumbaDeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning) # remember update df.append to pd.concat

class CellWithNetworkx:
    def __init__(self, swc_file):
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
        self.G = None # undirected graph
        self.DiG = None # directed graph
        self.sp = None
        self.distance_matrix = None

        self.num_syn_basal_exc = 0
        self.num_syn_apic_exc = 0
        self.num_syn_basal_inh = 0
        self.num_syn_apic_inh = 0
        self.num_syn_clustered = 0

        # we should have 2 rnd, the one for positioning should be fixed through the simu
        # while the one for generating spikes should be different for each simu
        self.rnd = np.random.RandomState(10) 
        self.spike_interval = 2000/1 # interval=2000(ms)/f
        
        self.syn_param_exc = [0, 0.3, 1.8, 0.0016] # reverse_potential, tau1, tau2, syn_weight
        self.syn_param_inh = [-86, 1, 8, 0.0008]

        self.sections_basal = [i for i in map(list, list(self.complex_cell.basal))] 
        self.sections_apical = [i for i in map(list, list(self.complex_cell.apical))]
        self.all_sections = [i for i in map(list, list(self.complex_cell.soma))] + self.sections_basal + self.sections_apical   

        self.section_df = pd.DataFrame(columns=['parent_id', 
                                                'section_id', 
                                                'parent_name', 
                                                'section_name', 
                                                'length', 
                                                'section_type'])
                                               
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
                                        
    
        self.spike_counts_basal_inh = None
        self.spike_counts_apic_inh = None
    
        self.time_interval = 1/2000 # 2ms, 0.002s
        self.FREQ_INH = 10  # Hz, /s
        self.DURATION = 2000

        # For clustered synapses

        # self.num_syn_clustered = None
        self.num_clusters = None
        self.cluster_radius = None
        self.distance_to_soma = None
        self.num_conn_per_preunit = None

        self.ori_dg_list = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        self.pref_ori_dg = None

        self.unit_ids = None
        self.indices = None

        # For tuning curve
        # self.num_spikes_list = []
        self.num_spikes_df = None 

        self.lock = None

        self.type_array = None

        self.class_dict = None
        self._create_graph()
        self._set_graph_order()

    # @jit(forceobj=True)
    def add_synapses(self, num_syn_basal_exc, num_syn_apic_exc, num_syn_basal_inh, num_syn_apic_inh):
        self.num_syn_basal_exc = num_syn_basal_exc
        self.num_syn_apic_exc = num_syn_apic_exc
        self.num_syn_basal_inh = num_syn_basal_inh
        self.num_syn_apic_inh = num_syn_apic_inh
        
        # add excitatory synapses
        self._add_single_synapses(num_syn_basal_exc, 'basal', 'exc')
        self._add_single_synapses(num_syn_apic_exc, 'apical', 'exc')
        
        # add inhibitory synapses
        self._add_single_synapses(num_syn_basal_inh, 'basal', 'inh')        
        self._add_single_synapses(num_syn_apic_inh, 'apical', 'inh')

    def assign_clustered_synapses(self, num_clusters, cluster_radius, distance_to_soma, num_conn_per_preunit=3):
        indices = self._generate_indices(num_clusters, num_conn_per_preunit)
        self.indices = indices
        
        self.num_clusters = num_clusters
        self.cluster_radius = cluster_radius
        self.distance_to_soma = distance_to_soma
        self.num_conn_per_preunit = num_conn_per_preunit

        # sec_syn_bg_exc_df = self.section_synapse_df[self.section_synapse_df['type'] == 'A']

        dist_list = self.class_dict.get(distance_to_soma, [])
        # sections directly connected to soma
        sections_ordered = [section[0].sec for i, section in enumerate(self.all_sections) if i in dist_list]
        
        for i in range(num_clusters):
            sec_syn_bg_exc_df = self.section_synapse_df[self.section_synapse_df['type'] == 'A']

            sec_syn_bg_exc_ordered_df = self.section_synapse_df[
                (self.section_synapse_df['section_synapse'].isin(sections_ordered)) & 
                (self.section_synapse_df['type'] == 'A')]

            index_list = indices[i]
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
                if list(self.DiG.successors(syn_ctr_sec_id)) != []:
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

            cluster_member_index = self._distance_synapse_mark_compare(dis_syn_from_ctr, dis_mark_from_ctr)
            # assign the surround as clustered synapse
            self.section_synapse_df.loc[syn_surround_ctr.iloc[cluster_member_index].index, 'type'] = 'C'
            self.section_synapse_df.loc[syn_surround_ctr.iloc[cluster_member_index].index, 'cluster_id'] = i
            self.section_synapse_df.loc[syn_surround_ctr.iloc[cluster_member_index].index, 'pre_unit_id'] = index_list[1:]

            print('cluster_id: ', i)
            print(self.section_synapse_df[self.section_synapse_df['cluster_id'] == i]['segment_synapse'].values)

    def visualize_synapses(self, folder_path, title='Synapses'):
        s = h.PlotShape(False)
        self._recursive_plot(s, self.section_synapse_df['segment_synapse'].values)
        plt.title(title)
        
        file_path = os.path.join(folder_path, f'figure_synapses.png')
        plt.savefig(file_path)
        plt.close()
    
    # @jit(forceobj=True)
    def add_inputs(self, folder_path):
        
        ori_dg_list, unit_ids, num_stims = self.ori_dg_list, self.unit_ids, 2
        # 创建一个空的 DataFrame
        self.num_spikes_df = pd.DataFrame(index=range(1, num_stims + 1), columns=ori_dg_list)
        self.lock = threading.Lock()
        # with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        #     # futures = []
        #     # for ori_dg in ori_dg_list:
        #     #     futures.append(executor.submit(self._process_ori_dg, ori_dg, unit_ids, num_stims, folder_path))
        #     futures = [executor.submit(self._process_ori_dg, ori_dg, unit_ids, num_stims, folder_path) for ori_dg in ori_dg_list]
        #     # futures = [executor.submit(self._process_ori_dg, ori_dg_list, unit_ids, num_stims, folder_path)]

        #     for future in futures:
        #         future.result()

        # self.num_spikes_df.to_csv('./num_spikes_df_parallel.csv', encoding='utf-8', index=False)

        # self._add_background_exc_inputs()
        # self._add_background_inh_inputs()
        for ori_dg in ori_dg_list: 
            # stim_ids vary across different orientations
            stim_ids = self._get_stim_ids(ori_dg)

            for stim_id in stim_ids[:num_stims]: 
                spt_unit_list = self._create_vecstim(ori_dg, stim_id, unit_ids)
                self._add_background_exc_inputs()
                self._add_clustered_inputs(spt_unit_list)
                self._add_background_inh_inputs()

                stim_index = np.where(stim_ids == stim_id)[0][0] + 1
                self._run_simulation(ori_dg, stim_id, stim_index)

                for i in plt.get_fignums():
                    plt.figure(i)
                    file_path = os.path.join(folder_path, f'figure_{ori_dg}_{stim_id}_{i}.png')
                    plt.savefig(file_path)
                    plt.close(i)
            
        self.num_spikes_df.to_csv('./num_spikes_df_oarallel.csv', encoding='utf-8', index=False)
    
    def _process_ori_dg(self, ori_dg, unit_ids, num_stims, folder_path):

        # for ori_dg in ori_dg_list:
        stim_ids = self._get_stim_ids(ori_dg)
        for stim_id in stim_ids[:num_stims]:
            spt_unit_list = self._create_vecstim(ori_dg, stim_id, unit_ids)
            self._add_background_exc_inputs()
            self._add_clustered_inputs(spt_unit_list)
            self._add_background_inh_inputs()

            stim_index = np.where(stim_ids == stim_id)[0][0] + 1
            self._run_simulation(ori_dg, stim_id, stim_index)

            with threading.Lock():
                for i in plt.get_fignums():
                    plt.figure(i)
                    file_path = os.path.join(folder_path, f'figure_{ori_dg}_{stim_id}_{i}.png')
                    plt.savefig(file_path)
                    plt.close(i)

    def visualize_tuning_curve(self):
        # num_spikes = self.num_spikes_list
        # pref_ori_dg = self.pref_ori_dg

        # x_values = np.array(ori_dg_list)
        # y_values = np.array(num_spikes)

        # # Fit Gaussian curve
        # # params = self.fit_gaussian(x_values, y_values)

        # # Plot original data points
        # plt.figure(figsize=(5, 5))
        # plt.plot(x_values, y_values, label='Data Points')

        # # plt.scatter(x_values, y_values, label='Data Points')

        # # Plot Gaussian curve using the fitted parameters
        # # curve_x = np.linspace(min(x_values), max(x_values), 100)
        # # curve_y = self.gaussian_function(curve_x, *params)
        # # plt.plot(curve_x, curve_y, label='Gaussian Fit', color='red')

        # # # Mark the data points on the graph
        # # for i, txt in enumerate(num_spikes):
        # #     plt.annotate(txt, (x_values[i], y_values[i]), textcoords="offset points", xytext=(0, 5), ha='center')

        # # Display the plot
        
        # plt.figure(figsize=(5, 5))
        # plt.xlabel('Orientation (degrees)')
        # plt.ylabel('Number of Spikes')
        # plt.legend()  

        # 假设有八个角度和每个角度对应的15个spike个数的值
        ori_dg_list = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        # spike_counts = np.random.randint(10, 20, size=(8, 15))  # 示例中使用随机生成的数据
        spike_counts = np.array(self.num_spikes_df)
        # 计算每个角度的平均值和标准差
        mean_spike_counts = np.mean(spike_counts, axis=1)
        std_dev_spike_counts = np.std(spike_counts, axis=1)

        # 绘制调谐曲线
        plt.errorbar(ori_dg_list, mean_spike_counts, yerr=std_dev_spike_counts, fmt='o-', capsize=5, label='Tuning Curve')

        # 在图上标出15个值对应的散点
        for ori_dg, spike_count_values in zip(ori_dg_list, spike_counts):
            plt.scatter([ori_dg] * len(spike_count_values), spike_count_values, color='gray', alpha=0.5)

        # 添加标签和标题
        plt.xlabel('Orientation (degrees)')
        plt.ylabel('Average Spike Count')
        plt.title('Tuning Curve with Scatter Points')
        plt.legend()
    
    def _run_simulation(self, ori_dg, stim_id, stim_index):
        soma_v = h.Vector().record(self.complex_cell.soma[0](0.5)._ref_v)
        dend_v = h.Vector().record(self.complex_cell.dend[0](0.5)._ref_v)
        apic_v = h.Vector().record(self.complex_cell.apic[0](0.5)._ref_v)
        time_v = h.Vector().record(h._ref_t)
        
        netcons_list = self.section_synapse_df['netcon']
        spike_times = [h.Vector() for _ in netcons_list]

        clustered_syn_index = self.section_synapse_df[self.section_synapse_df['type'] == 'C'].index
        spike_times_clustered = [spike_times[i] for i in clustered_syn_index]

        for nc, spike_times_vec in zip(netcons_list, spike_times):
            nc.record(spike_times_vec)

        h.tstop = self.DURATION
        st = time.time()
        h.run()
        print('complex cell simulation time {:.4f}'.format(time.time()-st))

        # 累加多个spike trains得到总的放电次数
        total_spikes = np.zeros(self.DURATION)  # 初始化总spike数数组

        for spike_times_vec in spike_times:
            try:
                total_spikes[np.floor(spike_times_vec).astype(int)] += 1 # 向下取整
            except ValueError:
                continue

        # 计算每个时间点的平均firing rate (Hz, /s)
        firing_rates = total_spikes / (np.mean(total_spikes) * self.time_interval)

        # threshold to define spikes
        threshold = 0

        num_spikes = self._count_spikes(soma_v, threshold)
        print(str(ori_dg)+'-'+str(stim_id))
        print("Number of spikes:", num_spikes)
        
        self.num_spikes_df.at[stim_index, ori_dg] = num_spikes
        # self.num_spikes_list.append(num_spikes)

        visualization_params = [ori_dg, stim_id, soma_v, dend_v, apic_v, time_v, spike_times_clustered]
        self._visualize_simulation(visualization_params)
        
    def _create_graph(self):
        all_sections = self.all_sections
        
        max_string_length = 50
    
        parent_list, parent_index_list = [], []
       
        for i, section_segment_list in enumerate(all_sections):
            section = section_segment_list[0].sec
            section_id = i
            section_name = section.psection()['name']
            match = re.search(r'\.(.*?)\[', section_name)
            section_type = match.group(1)
            L = section.psection()['morphology']['L']

            parent_list.append(section_name)
            parent_index_list.append(section_id)

            if i == 0:
                parent_name = 'None'
                parent_id = 0
                
            else:
                parent = section.psection()['morphology']['parent'].sec
                parent_name = parent.psection()['name']
                parent_id = parent_index_list[parent_list.index(parent_name)]
            
            # create data
            data_to_append = {'parent_id': parent_id,
                    'section_id': section_id,
                    'parent_name': parent_name,
                    'section_name': section_name,
                    'length': L,
                    'section_type': section_type}

            # self.section_df = self.section_df.append(data_to_append, ignore_index=True)
            self.section_df = pd.concat([self.section_df, pd.DataFrame(data_to_append, index=[0])], ignore_index=True)
            
        self.section_df.to_csv("cell1.csv", encoding='utf-8', index=False)
        Data = open('cell1.csv', "r")
        next(Data, None)  # skip the first line in the input file

        Graphtype = nx.Graph()
        DiGraphtype = nx.DiGraph()
        self.G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
                                  nodetype=int, data=(('parent_name', str), ('section_name', str),
                                                      ('length', float), ('section_type', str)))
        Data = open('cell1.csv', "r")
        next(Data, None)  # skip the first line in the input file
        self.DiG = nx.parse_edgelist(Data, delimiter=',', create_using=DiGraphtype,
                                     nodetype=int, data=(('parent_name', str), ('section_name', str),
                                                      ('length', float), ('section_type', str)))
        self.sp = dict(nx.all_pairs_shortest_path(self.G))

    def _set_graph_order(self):
        order_dict = nx.single_source_shortest_path_length(self.G, 0)

        # 创建一个空字典来保存分类结果
        self.class_dict = {}

        # 将每个点根据距离分类
        for node, order in order_dict.items():
            if order not in self.class_dict:
                self.class_dict[order] = []
            self.class_dict[order].append(node)

        # 获取最远的点到soma的距离（k值）
        max_order = max(order_dict.values())

        # 输出分类结果
        # for i in range(max_order + 1):
        #     print(f"Class {i}: {self.class_dict.get(i, [])}")

    def _add_single_synapses(self, num_syn, region, sim_type):
        sections = self.sections_basal if region == 'basal' else self.sections_apical
        e_syn, tau1, tau2, syn_weight = self.syn_param_exc if sim_type == 'exc' else self.syn_param_inh
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
            
            synapse = h.Exp2Syn(segment_synapse)
            synapse.e = e_syn
            synapse.tau1 = tau1
            synapse.tau2 = tau2

            data_to_append = {'section_id_synapse': section_id_synapse,
                            'section_synapse': section,
                            'segment_synapse': segment_synapse,
                            'synapse': synapse, 
                            'netstim': None,
                            'random': None,
                            'netcon': None,
                            'loc': loc,
                            'type': type,
                            'cluster_id': -1,
                            'pre_unit_id': -1,
                            'region': region}

            # with threading.Lock():
            with self.lock:
                self.section_synapse_df = self.section_synapse_df.append(data_to_append, ignore_index=True)

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            list(tqdm(executor.map(generate_synapse, range(num_syn)), total=num_syn))

        # for _ in tqdm(range(num_syn)):
        # # with tqdm(total=num_syn) as pbar:
        #     # i = 0
        #     # while i < num_syn:
        #     # section = self.rnd.choice(sections)[0].sec
        #     section = random.choices(sections, weights=section_length)[0][0].sec
        #     section_name = section.psection()['name']
            
        #     # Control the synpase density
        #     # max_num_syn = self.section_df.loc[self.section_df['section_name'] == section_name, 'max_num_syn'].values[0]
        #     # added_num_syn = self.section_df.loc[self.section_df['section_name'] == section_name, 'added_num_syn'].values[0]
            
        #     # if added_num_syn < max_num_syn:
        #     #     self.section_df.loc[self.section_df['section_name'] == section_name, 'added_num_syn'] += 1
        #     #     i += 1
        #     # else:
        #     #     continue

        #     section_id_synapse = self.section_df.loc[self.section_df['section_name'] == section_name, 'section_id'].values[0]

        #     loc = self.rnd.uniform()
        #     segment_synapse = section(loc)
            
        #     synapse = h.Exp2Syn(segment_synapse)
        #     synapse.e = e_syn
        #     synapse.tau1 = tau1
        #     synapse.tau2 = tau2

        #     data_to_append = {'section_id_synapse': section_id_synapse,
        #                                 'section_synapse': section,
        #                                 'segment_synapse': segment_synapse,
        #                                 'synapse': synapse, 
        #                                 'netstim': None,
        #                                 'random': None,
        #                                 'netcon': None,
        #                                 'loc': loc,
        #                                 'type': type,
        #                                 'cluster_id': -1,
        #                                 'pre_unit_id': -1,
        #                                 'region': region}
                
        #     self.section_synapse_df = self.section_synapse_df.append(data_to_append, ignore_index=True)
        #     # self.section_synapse_df = pd.concat([self.section_synapse_df, pd.DataFrame(data_to_append, index=[0])], ignore_index=True)

        #     # pbar.update(1)
        #     time.sleep(0.01)

    def _distance_synapse_mark_compare(self, dis_syn_from_ctr, dis_mark_from_ctr):
        # 创建一个包含原始索引的列表
        original_indices = list(range(len(dis_syn_from_ctr)))
        index = []

        for value in dis_mark_from_ctr:
            # 计算与value差值最小的元素的索引
            min_index = min(original_indices, key=lambda i: abs(dis_syn_from_ctr[i] - value)) 
            # 将该索引加入结果列表，并从original_indices中移除
            index.append(min_index)
            original_indices.remove(min_index)
        
        return index

    def _generate_indices(self, num_clusters, num_conn_per_preunit=3):
        
        spt_path = 'C:/Users/Windows/Desktop/MIMOlab/Codes/AllenAtlas/results/spt_df'
        pref_ori_dg = 0
        self.pref_ori_dg = pref_ori_dg
        session_id = 732592105
        ori_dg = 0.0

        # for calculate the OSI
        spt_file = glob.glob(spt_path + f'/{pref_ori_dg}_{session_id}/spt_{ori_dg}.csv')
        file_path = spt_file[0] # usually only one file
        spt_df = pd.read_csv(file_path, index_col=None, header=0)
        
        # we need the presynaptic units always the same
        unit_ids = np.sort(spt_df['unit_id'].unique())
        self.unit_ids = unit_ids

        results = []  # 用于存储生成的列表
        indices = []
        for _ in range(len(unit_ids)):
            # choose 3 clusters without replacement
            sampled = self.rnd.choice(num_clusters, num_conn_per_preunit, replace=False)  
            results.append(sampled)

        # 查找包含从0到k-1的列表的索引
        for i in range(num_clusters):
            index_list = [j for j, lst in enumerate(results) if i in lst]
            indices.append(index_list)

        return indices
    
    def _get_stim_ids(self, ori_dg):
        spt_path = 'C:/Users/Windows/Desktop/MIMOlab/Codes/AllenAtlas/results/spt_df'
        pref_ori_dg = 0
        self.pref_ori_dg = pref_ori_dg
        session_id = 732592105

        # for calculate the OSI
        spt_file = glob.glob(spt_path + f'/{pref_ori_dg}_{session_id}/spt_{ori_dg}.csv')
        file_path = spt_file[0] # usually only one file
        spt_df = pd.read_csv(file_path, index_col=None, header=0)
        
        # we need the presynaptic units always the same
        stim_ids = np.sort(spt_df['stimulus_presentation_id'].unique())
        print(f'stim_ids: {stim_ids}')
        
        return stim_ids

    def _add_background_exc_inputs(self):
        sec_syn_bg_exc_df = self.section_synapse_df[self.section_synapse_df['type'] == 'A']
        num_syn_background_exc = len(sec_syn_bg_exc_df)
        syn_weight = self.syn_param_exc[-1]

        def process_section(i):
            section = sec_syn_bg_exc_df.iloc[i]

            netstim = h.NetStim()
            netstim.interval = self.spike_interval
            netstim.number = 10
            netstim.start = 0
            netstim.noise = 1

            random = h.Random()
            random.Random123(i)
            random.negexp(1)
            netstim.noiseFromRandom(random)

            if section['netcon'] is not None:
                section['netcon'].weight[0] = 0

            synapse = section['synapse']
            netcon = h.NetCon(netstim, synapse)
            netcon.delay = 0
            netcon.weight[0] = syn_weight

            with self.lock:
                self.section_synapse_df.at[section.name, 'netstim'] = netstim
                self.section_synapse_df.at[section.name, 'random'] = random
                self.section_synapse_df.at[section.name, 'netcon'] = netcon

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            list(tqdm(executor.map(process_section, range(num_syn_background_exc)), total=num_syn_background_exc))

        # for i in tqdm(range(num_syn_background_exc)):
        #     # need this change updated to the global dataframe
        #     section = sec_syn_bg_exc_df.iloc[i]
            
        #     netstim = h.NetStim()
        #     netstim.interval = self.spike_interval
        #     netstim.number = 10
        #     netstim.start = 0
        #     netstim.noise = 1

        #     random = h.Random()
        #     random.Random123(i)
        #     random.negexp(1)
        #     netstim.noiseFromRandom(random)

        #     ## turn off the old netcons
        #     if section['netcon'] is not None:
        #         section['netcon'].weight[0] = 0

        #     synapse = section['synapse']
        #     netcon = h.NetCon(netstim, synapse) # netstim is always from the same unit with diff orientation
        #     netcon.delay = 0
        #     netcon.weight[0] = syn_weight
            
        #     self.section_synapse_df.at[section.name, 'netstim'] = netstim
        #     self.section_synapse_df.at[section.name, 'random'] = random
        #     self.section_synapse_df.at[section.name, 'netcon'] = netcon

        #     time.sleep(0.01)
    
    def _add_background_inh_inputs(self):  
        # sec_syn_exc_df = self.section_synapse_df[self.section_synapse_df['type'] == 'A' or self.section_synapse_df['type'] == 'C']
        exc_types = ['A', 'C']
        sec_syn_exc_df = self.section_synapse_df[self.section_synapse_df['type'].isin(exc_types)]

        exc_netcons_list = sec_syn_exc_df['netcon']
        syn_weight = self.syn_param_inh[-1]

        spike_times = [h.Vector() for _ in exc_netcons_list]
        for nc, spike_times_vec in zip(exc_netcons_list, spike_times):
            nc.record(spike_times_vec)

        h.tstop = self.DURATION
        st = time.time()
        h.run()
        print('complex cell simulation time {:.4f}'.format(time.time()-st))

        total_spikes = np.zeros(self.DURATION)  # 初始化总spike数数组

        for spike_times_vec in spike_times:
            try:
                total_spikes[np.floor(spike_times_vec).astype(int)] += 1 # 向下取整
            except ValueError:
                continue

        # 计算每个时间点的平均firing rate (Hz, /s)
        firing_rates = total_spikes / (np.mean(total_spikes) * self.time_interval)
        firing_rates_inh = firing_rates * self.FREQ_INH / np.mean(firing_rates)
        lambda_array = firing_rates_inh * self.time_interval
        
        self.spike_counts_basal_inh = np.random.poisson(lambda_array, size=(self.num_syn_basal_inh, self.DURATION))
        self.spike_counts_apic_inh = np.random.poisson(lambda_array, size=(self.num_syn_apic_inh, self.DURATION))

        sec_syn_bg_inh_df = self.section_synapse_df[self.section_synapse_df['type'] == 'B']
        
        def process_section(i):
            section = sec_syn_inh_df.iloc[i]

            counts = spike_counts_inh[i]
            spike_train = np.where(counts >= 1)[0] + 1000 * self.time_interval * np.random.rand(np.sum(counts >= 1))
            netstim = h.VecStim()
            netstim.play(h.Vector(spike_train))

            if section['netcon'] is not None:
                section['netcon'].weight[0] = 0

            synapse = section['synapse']
            netcon = h.NetCon(netstim, synapse)
            netcon.delay = 0
            netcon.weight[0] = syn_weight

            with self.lock:
                self.section_synapse_df.at[section.name, 'netstim'] = netstim
                self.section_synapse_df.at[section.name, 'netcon'] = netcon

        for region in ['basal', 'apical']:

            spike_counts_inh = self.spike_counts_basal_inh if region == 'basal' else self.spike_counts_apic_inh
            num_syn_background_inh = self.num_syn_basal_inh if region == 'basal' else self.num_syn_apic_inh
            sec_syn_inh_df = sec_syn_bg_inh_df[sec_syn_bg_inh_df['region'] == region]

            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                list(tqdm(executor.map(process_section, range(num_syn_background_inh)), total=num_syn_background_inh))

            # for i in tqdm(range(num_syn_background_inh)):
            #     section = sec_syn_inh_df.iloc[i]

            #     counts = spike_counts_inh[i]
            #     spike_train = np.where(counts>=1)[0] + 1000*self.time_interval*np.random.rand(np.sum(counts>=1)) 
            #     netstim = h.VecStim()    
            #     netstim.play(h.Vector(spike_train))

            #     ## turn off the old netcons
            #     if section['netcon'] is not None:
            #         section['netcon'].weight[0] = 0

            #     synapse = section['synapse']
            #     netcon = h.NetCon(netstim, synapse) # netstim is always from the same unit with diff orientation
            #     netcon.delay = 0
            #     netcon.weight[0] = syn_weight
                
            #     self.section_synapse_df.at[section.name, 'netstim'] = netstim
            #     self.section_synapse_df.at[section.name, 'netcon'] = netcon

            #     time.sleep(0.01)
    
    def _add_clustered_inputs(self, spt_unit_list):  
        sec_syn_clustered_df = self.section_synapse_df[self.section_synapse_df['type'] == 'C']
        num_syn_clustered = len(sec_syn_clustered_df)
        syn_weight = self.syn_param_exc[-1]

        for i in tqdm(range(num_syn_clustered)):
            # need this change updated to the global dataframe
            section = sec_syn_clustered_df.iloc[i]
            
            spt_unit = spt_unit_list[section['pre_unit_id']]
            # spt_unit_summed_list = spt_unit_summed_lists[section['cluster_id']]
            spt_unit_vector = h.Vector(spt_unit)
            netstim = h.VecStim()
            netstim.play(spt_unit_vector)

            ## turn off the old netcons
            if section['netcon'] is not None:
                section['netcon'].weight[0] = 0

            synapse = section['synapse']
            netcon = h.NetCon(netstim, synapse) # netstim is always from the same unit with diff orientation
            netcon.delay = 0
            netcon.weight[0] = syn_weight
            
            self.section_synapse_df.at[section.name, 'netstim'] = netstim
            self.section_synapse_df.at[section.name, 'netcon'] = netcon

            time.sleep(0.01)

        # self.visualize_simulation(1)
    
    def _create_vecstim(self, ori_dg, stim_id, unit_ids):
        # define the path of all csv files for spike trains
        spt_path = 'C:/Users/Windows/Desktop/MIMOlab/Codes/AllenAtlas/results/spt_df'

        # use glob.glob to extract the csv files with the orientation wanted
        # read every file in this folder
        pref_ori_dg = 0
        session_id = 732592105

        spt_file = glob.glob(spt_path + f'/{pref_ori_dg}_{session_id}/spt_{ori_dg}.csv')

        # folder_path = f'/{pref_ori_dg}_{session_id}/spt_{ori_dg}.csv'
        # spt_file = os.path.join(spt_path, folder_path)

        # for file_path in spt_file:
        file_path = spt_file[0] # usually only one file
        spt_df = pd.read_csv(file_path, index_col=None, header=0)
        
        ## Following part will differ across different ori_dg
        
        # not all units spike at each stimulus presentation, 
        # we need choose the stim_id with the max number of units 
        spt_grouped_df = spt_df.groupby(['unit_id', 'stimulus_presentation_id'])
        # max_stim_id = spt_df.groupby('stimulus_presentation_id')['unit_id'].nunique().idxmax()
        
        spt_unit_list = []
        ori_dg_list = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        for unit_id in unit_ids:
            try:
                spt_unit = spt_grouped_df.get_group((unit_id, stim_id))
                spt_unit = (spt_unit['spike_time'].values - spt_unit['spike_time'].values[0]) * 1000
            except KeyError:
                # for units not fired, add list of 0
                spt_unit = np.array([])

            spt_unit_list.append(spt_unit)

        return spt_unit_list
 
    def _visualize_simulation(self, visualization_params):
        ori_dg, stim_id, soma_v, dend_v, apic_v, time_v, spike_times_clustered = visualization_params
        
        # plotting the results
        plt.figure(figsize=(5, 5))
        plt.title(str(ori_dg) + '-' + str(stim_id))
        for i, spike_times_vec in enumerate(spike_times_clustered):
            try:
                if len(spike_times_vec) > 0:
                    plt.vlines(spike_times_vec, i + 0.5, i + 1.5)
            except IndexError:
                continue  

        # 绘制firing rate曲线
        # plt.figure(figsize=(5, 5))
        # plt.plot(firing_rates,color='blue',label='Backgournd Excitatory Firing Rate')
        # plt.legend()
        # plt.xlabel('Time(ms)')
        # plt.ylabel('Firing Rate(Hz)')
        # plt.title('Firing Rate Curve')

        # 使用高斯核进行卷积，得到平滑的firing rate曲线 (don't consider currently)
        # sigma = 1  # 高斯核的标准差
        # smoothed_firing_rates = gaussian_filter1d(firing_rates, sigma)

        # plt.figure(figsize=(5, 5))
        # plt.plot(smoothed_firing_rates, label='Smoothed Firing Rate')
        # plt.legend()
        # plt.xlabel('Time(ms)')
        # plt.ylabel('Firing Rate(Hz)')
        # plt.title('Firing Rate Curve')

        plt.figure(figsize=(5, 5))
        plt.plot(time_v, soma_v, label='soma')
        plt.plot(time_v, dend_v, label='basal')
        plt.plot(time_v, apic_v, label='apical')
        plt.legend()
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        plt.title(str(ori_dg) + '-' + str(stim_id))

    def _count_spikes(self, soma_voltage, threshold=0):
        spike_count = 0
        is_spiking = False

        for voltage in soma_voltage:
            if voltage > threshold and not is_spiking:
                is_spiking = True
                spike_count += 1
            elif voltage <= threshold:
                is_spiking = False

        return spike_count

    def calculate_cirvar(self, num_spikes, ori_dg_list):
        # 将角度转换为弧度
        theta_radians = np.deg2rad(ori_dg_list)

        # 计算 exp(2i * theta)
        complex_exp = np.exp(2j * theta_radians)

        # 计算加权和
        weighted_sum = np.sum(num_spikes * complex_exp) / np.sum(num_spikes)

        # 取模
        cirvar = round(np.abs(weighted_sum), 4)

        return cirvar

    def calculate_osi(self, num_spikes, ori_dg_list, pref_ori_dg):
        
        r_pref = num_spikes[ori_dg_list.index(pref_ori_dg)] + num_spikes[ori_dg_list.index((pref_ori_dg + 180) % 360)]
        r_ortho = num_spikes[ori_dg_list.index((pref_ori_dg + 90)) % 360] + num_spikes[ori_dg_list.index((pref_ori_dg + 270) % 360)]

        osi = round((r_pref - r_ortho) / (r_pref + r_ortho), 4)
        
        return osi

    def gaussian_function(self, x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

    def fit_gaussian(self, x, y):
        popt, _ = curve_fit(self.gaussian_function, x, y, p0=[1, np.mean(x), np.std(x)])
        return popt

    def _recursive_plot(self, s, seg_list, index=0):
        markers = {
            'A': 'or',  # 红色圆形
            'B': 'xb',  # 蓝色叉形
            'C': 'sg',  # 绿色方形
            'D': 'pk',  # 紫色五角星
            'E': 'dc',  # 青色钻石形
            'F': '^m',  # 品红色三角形
            'G': '*y',  # 黄色星型
            'H': '+k',  # 黑色十字形 
        }

        if index == 0:
            return self._recursive_plot(s.plot(plt), seg_list, index+1)
        elif index <= len(seg_list):
            # if self.initialize_cluster_flag == False:
            segment_type = self.type_array[index - 1]
            marker = markers.get(segment_type, 'or')  # 如果类型不在字典中，默认使用'or'作为标记
            return self._recursive_plot(s.mark(seg_list[index - 1], marker), seg_list, index + 1)
        
                # if self.type_array[index-1] == 'A':
                #     return self._recursive_plot(s.mark(seg_list[index-1],'or'), seg_list, index+1)
                # else:
                #     return self._recursive_plot(s.mark(seg_list[index-1],'xb'), seg_list, index+1)
            # else:
            #     return self._recursive_plot(s.mark(seg_list[index-1],'xr'), seg_list, index+1)
        
    def add_clustered_synapses(self, num_syn_clustered=50, num_clusters=5, cluster_radius=2.5, distance_to_soma=1):
        sections = self.sections_basal + self.sections_apical 
        all_sections = self.all_sections 

        self.distance_to_soma = distance_to_soma
        dist_list = self.class_dict.get(distance_to_soma, [])
        # sections directly connected to soma
        sections = [section for i, section in enumerate(all_sections) if i in dist_list]

        e_syn, tau1, tau2, syn_weight = self.syn_param_exc

        self.num_syn_clustered = num_syn_clustered
        self.num_clusters = num_clusters
        self.cluster_radius = cluster_radius
        num_syn = num_syn_clustered - num_clusters

        # points_per_cluster = np.ceil(np.random.normal(num_syn_clustered/num_clusters - 1, 2, num_clusters))
        # points_per_clutser = [max(int(round(i)),0) for i in points_per_cluster]

        section_cluster_list = []

        for i in tqdm(range(num_clusters)):
                
            # could add a new attribute, order, for each section; 
            # and then extract different number of sections from different orders
            # Section = self.rnd.choice(self.sections_basal+self.sections_apical)

            # Only select the sections directly connected to soma
            # section = self.rnd.choice(sections)[0].sec
            section = self.rnd.choice(sections)[0].sec

            section_cluster_list.append(section)

            section_name = section.psection()['name']
            section_id_synapse = self.section_df.loc[self.section_df['section_name'] == section_name, 'section_id'].values[0]
            
            # loc = self.rnd.uniform()
            loc = 0.5 # for center
            
            segment_synapse = section(loc)

            synapse = h.Exp2Syn(segment_synapse)
            synapse.e = e_syn
            synapse.tau1 = tau1
            synapse.tau2 = tau2

            data_to_append = {'section_id_synapse': section_id_synapse,
                                        'section_synapse': section,
                                        'segment_synapse': segment_synapse,
                                        'synapse': synapse, 
                                        'netstim': None,
                                        'random': None,
                                        'netcon': None,
                                        'loc': loc,
                                        'type': 'C',
                                        'cluster_id': i}
                
            self.section_synapse_df = self.section_synapse_df.append(data_to_append, ignore_index=True)    
            # self.section_synapse_df = pd.concat([self.section_synapse_df, pd.DataFrame(data_to_append, index=[0])], ignore_index=True)
            
        for _ in tqdm(range(num_syn)):
            # available_index_section_cluster_list = [i for i, count in enumerate(points_per_cluster) if count > 0]
            # available_section_cluster_list = [section_cluster_list[i] for i in available_index_section_cluster_list]
            # section_cluster = self.rnd.choice(available_section_cluster_list)
            section_cluster = self.rnd.choice(section_cluster_list)
            section_cluster_index = section_cluster_list.index(section_cluster)
            section_name_cluster = section_cluster.psection()['name']
            section_id_synapse_cluster = self.section_df.loc[self.section_df['section_name'] == section_name_cluster, 'section_id'].values[0]
            
            # use exponential distribution to generate loc
            dis_from_center = self.rnd.exponential(1/cluster_radius)

            if self.rnd.random() < 0.5:
                loc = 0.5 + dis_from_center / section_cluster.L  
            else:
                loc = 0.5 - dis_from_center / section_cluster.L
            
            start_point = 0.5
            
            while loc > 1 or loc < 0:
                if loc > 1:  
                    if list(self.DiG.successors(section_id_synapse_cluster)) != []:
                        dis_from_center = dis_from_center - section_cluster.L * (1 - start_point)
                        section_id_synapse_cluster = self.rnd.choice(list(self.DiG.successors(section_id_synapse_cluster)))
                        section_cluster = all_sections[section_id_synapse_cluster][0].sec
                        loc = dis_from_center / section_cluster.L
                        start_point = 0
                    else:
                        loc = 1
                elif loc < 0:
                    if list(self.DiG.predecessors(section_id_synapse_cluster)) != []:
                        dis_from_center = dis_from_center - section_cluster.L * (start_point - 0)
                        section_id_synapse_cluster = self.rnd.choice(list(self.DiG.predecessors(section_id_synapse_cluster)))
                        section_cluster = all_sections[section_id_synapse_cluster][0].sec
                        loc = 1 - dis_from_center / section_cluster.L
                        start_point = 1
                    else:
                        loc = 0
                    
            segment_synapse = section_cluster(loc)

            synapse = h.Exp2Syn(segment_synapse)
            synapse.e = e_syn
            synapse.tau1 = tau1
            synapse.tau2 = tau2

            data_to_append = {'section_id_synapse': section_id_synapse_cluster,
                                        'section_synapse': section_cluster,
                                        'segment_synapse': segment_synapse,
                                        'synapse': synapse, 
                                        'netstim': None,
                                        'random': None,
                                        'netcon': None,
                                        'loc': loc,
                                        'type': 'C',
                                        'cluster_id': section_cluster_index}
                
            self.section_synapse_df = self.section_synapse_df.append(data_to_append, ignore_index=True)
            # self.section_synapse_df = pd.concat([self.section_synapse_df, pd.DataFrame(data_to_append, index=[0])], ignore_index=True)
            
            time.sleep(0.01)

        self.type_array = self.section_synapse_df['type'].values

        # self.add_clustered_inputs(syn_weight, num_clusters)

    def set_synapse_type(self):
        rnd = self.rnd
        type_list = ['A', 'B']  
        num_clusters_per_type = (5, 11) 
        min_points = 10
        max_points = 100
        center_type_prob = 1  

        type_array = np.array([''] * self.num_syn, dtype=str)

        num_types = len(type_list)
        cluster_centers = []

        section_id_synapse_list = self.section_id_synapse_list
        distance_matrix = self.distance_matrix

        for i in range(num_types):
            num_clusters = np.random.randint(*num_clusters_per_type)
            cluster_centers_type = np.random.choice(section_id_synapse_list, num_clusters, replace=False)
            cluster_centers.extend([(center, type_list[i]) for center in cluster_centers_type])

        for centers, ptype in cluster_centers:
            idx = np.where(section_id_synapse_list == centers)[0]
            type_array[idx] = ptype

        while '' in type_array:
            for centers, ptype in cluster_centers:
                num_points = np.random.randint(min_points, max_points + 1)
                if np.count_nonzero(type_array == '') < self.num_syn / 2:
                    num_points = np.random.randint(min_points, min(max_points, 31))  
                distances = distance_matrix[:, centers]
                eligible_indices = np.where(type_array == '')[0]
                nearest_indices = eligible_indices[np.argsort(distances[eligible_indices])[:num_points]]
                
                # type_array[nearest_indices] = ptype
                non_center_types = [t for t in type_list if t != ptype]
                chosen_type = rnd.choice([ptype] + non_center_types, p=[center_type_prob] + [(1 - center_type_prob) / (num_types - 1)] * (num_types - 1), size=len(nearest_indices))
                type_array[nearest_indices] = chosen_type

        self.type_array = type_array
    
    #cannot use jit for this function
    # @jit
    def calculate_distance_matrix(self, distance_limit=2000):
        loc_array, section_id_synapse_list = self.loc_array, self.section_id_synapse_list
        parentID_list, length_list = self.section_df['parent_id'].values, self.section_df['length'].values
        
        length_list = np.array(length_list)
        distance_matrix = np.zeros((self.num_syn, self.num_syn))
        for i in range(self.num_syn):
            for j in range(i + 1, self.num_syn):
                    m = section_id_synapse_list[i]
                    n = section_id_synapse_list[j]

                    path = self.sp[m][n]

                    if len(path) > 1:
                        loc_i = loc_array[i] * (parentID_list[m] == path[1]) + (1-loc_array[i]) * (parentID_list[m] != path[1])
                        loc_j = loc_array[j] * (parentID_list[n] == path[-2]) + (1-loc_array[j]) * (parentID_list[n] != path[-2])
                        
                        mask_i = np.array(path) == m
                        mask_j = np.array(path) == n

                        distance = np.sum(length_list[path] * (mask_i * loc_i + mask_j * loc_j))
                        # for k in path:
                        #     if k == m:
                        #         distance = length_list[k]*loc_i
                        #     if k == n:
                        #         distance = length_list[k]*loc_j
                        #     distance = distance + length_list[k]
                    else:
                        distance = length_list[m] * abs(loc_array[i] - loc_array[j])

                    distance_matrix[i, j] = distance_matrix[j, i] = distance

        self.distance_matrix = distance_matrix

        return distance_matrix

    def visualize_distance(self):
        type_array, distance_matrix = self.section_synapse_df['type'].values, self.distance_matrix
        
        distance_list, distance_a_a_list, distance_b_b_list = [], [], []
        for i in range(self.num_syn):
            for j in range(self.num_syn):
                if i < j:
                    if self.initialize_cluster_flag == False:
                        distance_list.append(distance_matrix[i,j])
                        if type_array[i] == type_array[j] == 'A' :#and distance_matrix[i,j] < distance_limit:
                            distance_a_a_list.append(distance_matrix[i,j])
                        if type_array[i] == type_array[j] == 'B' :#and distance_matrix[i,j] < distance_limit:
                            distance_b_b_list.append(distance_matrix[i,j])
                    else:
                        distance_list.append(distance_matrix[i,j])
                    
        plt.figure(figsize=(5, 5))
        if self.initialize_cluster_flag == False:
            sns.histplot(distance_a_a_list, kde=True,stat="density",color='lightskyblue',linewidth=0,label='A-A')
            sns.histplot(distance_b_b_list,kde=True,stat="density",color='orange',linewidth=0,label='B-B')
        sns.histplot(distance_list,kde=True,stat="density",color='lightgreen',linewidth=0,label='All')
        plt.legend()
        plt.xlabel('Distance (microns)')
        plt.ylabel('Probability')
        # plt.xlim(0,distance_limit)
        # plt.ylim(0,0.004)
        plt.title('Distance distribution before clustered')

    def set_type_array(self, new_type_array):
        self.type_array = new_type_array

    def get_cell(self):
        return self.complex_cell

