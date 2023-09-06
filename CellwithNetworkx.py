from neuron import gui, h
from neuron.units import ms, mV
import numpy as np
import time
import pandas as pd
import networkx as nx
from tqdm import tqdm
import warnings
import re
from math import floor
import matplotlib.pyplot as plt
import seaborn as sns
import numba 
from numba import jit
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=numba.NumbaDeprecationWarning)

class CellwithNetworkx:
    def __init__(self, swc_file):
        h.load_file("import3d.hoc")
        h.nrn_load_dll('./mod/nrnmech.dll')
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

        self.rnd = np.random.RandomState(10)
        self.spike_interval = 1000/1 # interval=1000(ms)/f
        
        self.syn_param_exc = [0, 0.3, 1.8, 0.0016]
        self.syn_param_inh = [-86, 1, 8, 0.0008]

        self.sections_basal = [i for i in map(list, list(self.complex_cell.basal))] 
        self.sections_apical = [i for i in map(list, list(self.complex_cell.apical))]
        self.all_sections = [i for i in map(list, list(self.complex_cell.soma))] + self.sections_basal + self.sections_apical   

        self.section_df = pd.DataFrame(columns=['section_id', 
                                                'parentID', 
                                                'section_name', 
                                                'parentName', 
                                                'length', 
                                                'sectionType'])

        self.section_synapse_df = pd.DataFrame(columns=['section_id_synapse',
                                                'section_synapse',
                                                'segment_synapse',
                                                'synapse', 
                                                'netstim',  
                                                'random',
                                                'netcon',
                                                'loc',
                                                'type'])
                                        
    
        self.spike_counts_basal_inh = None
        self.spike_counts_apic_inh = None
    
        self.time_interval = 1/1000 # 1ms, 0.001s
        self.FREQ_INH = 10  # Hz, /s
        self.DURATION = 1000

        # For clustered synapses
        self.num_syn_clustered = None
        self.k = None
        self.cluster_radius = None

        self.type_array = None

        self._create_graph()
        # self._set_graph_order()

    def _create_graph(self):
        all_sections = self.all_sections
        
        max_string_length = 50
        # Create DataFrame to store section information
        # section_df = 
        parent_list, parent_index_list = [], []
       
        for i, section in enumerate(all_sections):
            Section = section[0].sec
            section_id = i
            section_name = Section.psection()['name']
            match = re.search(r'\.(.*?)\[', section_name)
            sectionType = match.group(1)
            L = Section.psection()['morphology']['L']

            parent_list.append(section_name)
            parent_index_list.append(section_id)

            if i == 0:
                parentName = 'None'
                parentID = 0
                
            else:
                parent = Section.psection()['morphology']['parent'].sec
                parentName = parent.psection()['name']
                parentID = parent_index_list[parent_list.index(parentName)]
            
            # create data
            data_to_append = {'section_id': section_id,
                    'parentID': parentID,
                    'section_name': section_name,
                    'parentName': parentName,
                    'length': L,
                    'sectionType': sectionType}

            self.section_df = self.section_df.append(data_to_append, ignore_index=True)
            
        self.section_df.to_csv("cell1.csv", encoding='utf-8', index=False)
        Data = open('cell1.csv', "r")
        next(Data, None)  # skip the first line in the input file
        Graphtype = nx.Graph()
        DiGraphtype = nx.DiGraph()
        self.G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
                                  nodetype=int, data=(('section_name', str), ('parentName', str),
                                                      ('length', float), ('sectionType', str)))
        Data = open('cell1.csv', "r")
        next(Data, None)  # skip the first line in the input file
        self.DiG = nx.parse_edgelist(Data, delimiter=',', create_using=DiGraphtype,
                                     nodetype=int, data=(('section_name', str), ('parentName', str),
                                                      ('length', float), ('sectionType', str)))
        self.sp = dict(nx.all_pairs_shortest_path(self.G))

        # self.section_df = section_df
        # self.parentID_list, self.section_id_list, self.section_name_list, self.length_list = parentID_list, section_id_list, section_name_list, length_list
    
    def _set_graph_order(self):
        order_dict = nx.single_source_shortest_path_length(self.G, 0)

        # 创建一个空字典来保存分类结果
        class_dict = {}

        # 将每个点根据距离分类
        for node, order in order_dict.items():
            if order not in class_dict:
                class_dict[order] = []
            class_dict[order].append(node)

        # 获取最远的点到soma的距离（k值）
        max_order = max(order_dict.values())

        # 输出分类结果
        for i in range(max_order + 1):
            print(f"Class {i}: {class_dict.get(i, [])}")

    # @jit
    def add_background_synapses(self, num_syn_basal_exc, num_syn_apic_exc, num_syn_basal_inh, num_syn_apic_inh):
        self.num_syn_basal_exc = num_syn_basal_exc
        self.num_syn_apic_exc = num_syn_apic_exc
        self.num_syn_basal_inh = num_syn_basal_inh
        self.num_syn_apic_inh = num_syn_apic_inh
        
        self.add_synapses(num_syn_basal_exc, 'basal', 'exc')
        self.add_synapses(num_syn_apic_exc, 'apical', 'exc')
        
        netcons_list = self.section_synapse_df['netcon']
        spike_times = [h.Vector() for _ in netcons_list]
        for nc, spike_times_vec in zip(netcons_list, spike_times):
            nc.record(spike_times_vec)

        h.tstop = self.DURATION
        h.run()

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
        
        self.spike_counts_basal_inh = np.random.poisson(lambda_array, size=(num_syn_basal_inh, self.DURATION))
        self.spike_counts_apic_inh = np.random.poisson(lambda_array, size=(num_syn_apic_inh, self.DURATION))

        # 根据spike_counts生成spike trains  
        self.add_synapses(num_syn_basal_inh, 'basal', 'inh')        
        self.add_synapses(num_syn_apic_inh, 'apical', 'inh')

        self.visualize_simulation()

    def add_synapses(self, num_syn, region, sim_type):
        sections = self.sections_basal if region == 'basal' else self.sections_apical
        e_syn, tau1, tau2, syn_weight = self.syn_param_exc if sim_type == 'exc' else self.syn_param_inh
        
        if sim_type == 'exc':
            for _ in tqdm(range(num_syn)):
                section = self.rnd.choice(sections)[0].sec
                section_name = section.psection()['name']
                
                section_id_synapse = self.section_df.loc[self.section_df['section_name'] == section_name, 'section_id'].values[0]

                loc = self.rnd.uniform()
                segment_synapse = section(loc)
                
                synapse = h.Exp2Syn(segment_synapse)
                synapse.e = e_syn
                synapse.tau1 = tau1
                synapse.tau2 = tau2

                netstim = h.NetStim()
                netstim.interval = self.spike_interval
                netstim.number = 10
                netstim.start = 0
                netstim.noise = 1

                random = h.Random()
                try:
                    random_length = len(self.section_synapse_df['random'])
                except KeyError:
                    random_length = 0
                random.Random123(random_length)
                random.negexp(1)
                netstim.noiseFromRandom(random)

                netcon = h.NetCon(netstim, synapse)
                netcon.delay = 0
                netcon.weight[0] = syn_weight

                data_to_append = {'section_id_synapse': section_id_synapse,
                                            'section_synapse': section,
                                            'segment_synapse': segment_synapse,
                                            'synapse': synapse, 
                                            'netstim': netstim,
                                            'random': random,
                                            'netcon': netcon,
                                            'loc': loc,
                                            'type': 'A'}
                    
                self.section_synapse_df = self.section_synapse_df.append(data_to_append, ignore_index=True)
            
                time.sleep(0.01)
        else:
            spike_counts_inh = self.spike_counts_basal_inh if region == 'basal' else self.spike_counts_apic_inh
            
            for i in tqdm(range(num_syn)):
                section = self.rnd.choice(sections)[0].sec
                section_name = section.psection()['name']
                section_id_synapse = self.section_df.loc[self.section_df['section_name'] == section_name, 'section_id'].values[0]

                loc = self.rnd.uniform()
                segment_synapse = section(loc)

                synapse = h.Exp2Syn(segment_synapse)
                synapse.e = e_syn
                synapse.tau1 = tau1
                synapse.tau2 = tau2

                counts = spike_counts_inh[i]
                spike_train = np.where(counts>=1)[0] + 1000*self.time_interval*np.random.rand(np.sum(counts>=1)) 
                netstim = h.VecStim()    
                netstim.play(h.Vector(spike_train))

                netcon = h.NetCon(netstim, synapse)
                netcon.delay = 0
                netcon.weight[0] = syn_weight

                data_to_append = {'section_id_synapse': section_id_synapse,
                                            'section_synapse': section,
                                            'segment_synapse': segment_synapse,
                                            'synapse': synapse, 
                                            'netstim': netstim,
                                            'random': None,
                                            'netcon': netcon,
                                            'loc': loc,
                                            'type': 'B'}
                
                self.section_synapse_df = self.section_synapse_df.append(data_to_append, ignore_index=True)
                    
                time.sleep(0.01)
        
    def add_clustered_synapses(self, num_syn_clustered=50, k=5, cluster_radius=2.5):
        sections = self.sections_basal + self.sections_apical #[i for i in map(list, list(self.complex_cell.basal))] + [i for i in map(list, list(self.complex_cell.apical))]
        all_sections = self.all_sections #[i for i in map(list, list(self.complex_cell.soma))] + sections_basal_apical

        e_syn, tau1, tau2, syn_weight = self.syn_param_exc

        self.num_syn_clustered = num_syn_clustered
        self.k = k
        self.cluster_radius = cluster_radius
        num_syn = num_syn_clustered - k

        # points_per_cluster = np.ceil(np.random.normal(num_syn_clustered/k - 1, 2, k))
        # points_per_clutser = [max(int(round(i)),0) for i in points_per_cluster]

        section_cluster_list = []

        for _ in tqdm(range(k)):
            
            # could add a new attribute, order, for each section; 
            # and then extract different number of sections from different orders
            # Section = self.rnd.choice(self.sections_basal+self.sections_apical)
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

            netstim = h.NetStim()
            netstim.interval = self.spike_interval
            netstim.number = 1
            netstim.start = 0
            netstim.noise = 0
            
            netcon = h.NetCon(netstim, synapse)
            netcon.delay = 0
            netcon.weight[0] = syn_weight

            data_to_append = {'section_id_synaps': section_id_synapse,
                                        'section_synapse': section,
                                        'segment_synapse': segment_synapse,
                                        'synapse': synapse, 
                                        'netstim': netstim,
                                        'random': None,
                                        'netcon': netcon,
                                        'loc': loc,
                                        'type': 'C'}
                
            self.section_synapse_df = self.section_synapse_df.append(data_to_append, ignore_index=True)
            
        for _ in tqdm(range(num_syn)):
            # available_index_section_cluster_list = [i for i, count in enumerate(points_per_cluster) if count > 0]
            # available_section_cluster_list = [section_cluster_list[i] for i in available_index_section_cluster_list]
            # section_cluster = self.rnd.choice(available_section_cluster_list)
            section_cluster = self.rnd.choice(section_cluster_list)
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
                    
            # if loc > 1 or loc < 0: # (update this)
            #     gap = floor(loc)
            #     # loc = loc - gap
            #     dis_from_center = dis_from_center
            #     # Find the new section with Networkx (successors or predecessors)
            #     if gap < 0:
            #         if list(self.DiG.predecessors(section_id_synapse_cluster)) != []:
            #             section_id_synapse_cluster = next(list(self.DiG.predecessors(section_id_synapse_cluster))[0] for _ in range(abs(gap)))
            #         else:
            #             loc, gap = 0, 0
            #     elif gap > 0:
            #         if list(self.DiG.successors(section_id_synapse_cluster)) != []:
            #             section_id_synapse_cluster = next(list(self.DiG.successors(section_id_synapse_cluster))[0] for _ in range(gap))
            #         else:
            #             loc, gap = 1, 0

            #     section_cluster = all_sections[section_id_synapse_cluster][0].sec

            segment_synapse = section_cluster(loc)

            synapse = h.Exp2Syn(segment_synapse)
            synapse.e = e_syn
            synapse.tau1 = tau1
            synapse.tau2 = tau2

            netstim = h.NetStim()
            netstim.interval = self.spike_interval
            netstim.number = 1
            netstim.start = 0
            netstim.noise = 0
            
            netcon = h.NetCon(netstim, synapse)
            netcon.delay = 0
            netcon.weight[0] = syn_weight

            data_to_append = {'section_id_synapse': section_id_synapse_cluster,
                                        'section_synapse': section_cluster,
                                        'segment_synapse': segment_synapse,
                                        'synapse': synapse, 
                                        'netstim': netstim,
                                        'random': None,
                                        'netcon': netcon,
                                        'loc': loc,
                                        'type': 'C'}
                
            self.section_synapse_df = self.section_synapse_df.append(data_to_append, ignore_index=True)
            
            time.sleep(0.01)

        self.type_array = self.section_synapse_df['type'].values

        self.visualize_simulation()
    
    #cannot use jit for this function either
    # @jit
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
        parentID_list, length_list = self.section_df['parentID'].values, self.section_df['length'].values
        
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
            if self.initialize_cluster_flag == False:
                segment_type = self.type_array[index - 1]
                marker = markers.get(segment_type, 'or')  # 如果类型不在字典中，默认使用'or'作为标记
                return self._recursive_plot(s.mark(seg_list[index - 1], marker), seg_list, index + 1)
        
                # if self.type_array[index-1] == 'A':
                #     return self._recursive_plot(s.mark(seg_list[index-1],'or'), seg_list, index+1)
                # else:
                #     return self._recursive_plot(s.mark(seg_list[index-1],'xb'), seg_list, index+1)
            else:
                return self._recursive_plot(s.mark(seg_list[index-1],'xr'), seg_list, index+1)
        
    def visualize_synapses(self,title):
        s = h.PlotShape(False)
        self._recursive_plot(s, self.segment_synapse_list + self.segment_synapse_clustered_list)
        self._recursive_plot(s, self.segment_synapse_list + self.segment_synapse_clustered_list)
        plt.title(title)
 
    def visualize_simulation(self):
        soma_v = h.Vector().record(self.complex_cell.soma[0](0.5)._ref_v)
        dend_v = h.Vector().record(self.complex_cell.dend[0](0.5)._ref_v)
        apic_v = h.Vector().record(self.complex_cell.apic[0](0.5)._ref_v)
        time_v = h.Vector().record(h._ref_t)
        
        netcons_list = self.section_synapse_df['netcon']
        spike_times = [h.Vector() for _ in netcons_list]
        for nc, spike_times_vec in zip(netcons_list, spike_times):
            nc.record(spike_times_vec)

        h.tstop = self.DURATION
        st = time.time()
        h.run()
        print('complex cell simulation time {:.4f}'.format(time.time()-st))

        # plotting the results
        plt.figure(figsize=(5, 5))
        for i, spike_times_vec in enumerate(spike_times):
            try:
                if len(spike_times_vec) > 0:
                    plt.vlines(spike_times_vec, i + 0.5, i + 1.5)
            except IndexError:
                continue  

        # 累加多个spike trains得到总的放电次数
        total_spikes = np.zeros(self.DURATION)  # 初始化总spike数数组

        for spike_times_vec in spike_times:
            try:
                total_spikes[np.floor(spike_times_vec).astype(int)] += 1 # 向下取整
            except ValueError:
                continue

        # 计算每个时间点的平均firing rate (Hz, /s)
        firing_rates = total_spikes / (np.mean(total_spikes) * self.time_interval)

        # 绘制firing rate曲线
        plt.figure(figsize=(5, 5))
        plt.plot(firing_rates,color='blue',label='Backgournd Excitatory Firing Rate')
        plt.legend()
        plt.xlabel('Time(ms)')
        plt.ylabel('Firing Rate(Hz)')
        plt.title('Firing Rate Curve')

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

