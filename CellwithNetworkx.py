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
        self.numSyn = None

        self.rnd = np.random.RandomState(10)

        self.k = None
        self.cluster_radius = None
        self.initialize_cluster_flag = False

        self.parentID_list = None
        self.sectionID_list = None
        self.sectionName_list = None
        self.length_list = None

        self.loc_array = None
        self.type_array = None
        self.sectionID_synapse_list = None
        self.section_synapse_list = None
        self.segment_synapse_list = None

        self.cluster_radius = 2.5 # micron

        self.netcons_list = None

        self._create_graph()
        # self._set_graph_order()

    def _create_graph(self):
        all_sections = [i for i in map(list, list(self.complex_cell.soma))] + [i for i in map(list, list(self.complex_cell.basal))] + [i for i in map(list, list(self.complex_cell.apical))]

        max_string_length = 50
        # Create DataFrame to store section information
        df = pd.DataFrame(columns=['sectionID', 'parentID', 'sectionName', 'parentName', 'length'])

        parent_list, parentID_list, parent_index_list, sectionID_list, sectionName_list, length_list = [], [], [], [], [], []
       
        for i, section in enumerate(all_sections):
            Section = section[0].sec
            sectionID = i
            sectionName = Section.psection()['name']
            match = re.search(r'\.(.*?)\[', sectionName)
            sectionType = match.group(1)
            L = Section.psection()['morphology']['L']

            parent_list.append(sectionName)
            parent_index_list.append(sectionID)

            if i == 0:
                parentName = 'None'
                parentID = 0
                
            else:
                parent = Section.psection()['morphology']['parent'].sec
                parentName = parent.psection()['name']
                parentID = parent_index_list[parent_list.index(parentName)]

            sectionID_list.append(sectionID)
            sectionName_list.append(sectionName)
            parentID_list.append(parentID)
            length_list.append(L)
            
            # create data
            data = {'sectionID': sectionID,
                    'parentID': parentID,
                    'sectionName': sectionName,
                    'parentName': parentName,
                    'length': L,
                    'sectionType': sectionType}

            df = pd.concat([df, pd.DataFrame(data, index=[0])])

        df.to_csv("cell1.csv", encoding='utf-8', index=False)
        Data = open('cell1.csv', "r")
        next(Data, None)  # skip the first line in the input file
        Graphtype = nx.Graph()
        DiGraphtype = nx.DiGraph()
        self.G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
                                  nodetype=int, data=(('sectionName', str), ('parentName', str),
                                                      ('length', float), ('sectionType', str)))
        Data = open('cell1.csv', "r")
        next(Data, None)  # skip the first line in the input file
        self.DiG = nx.parse_edgelist(Data, delimiter=',', create_using=DiGraphtype,
                                     nodetype=int, data=(('sectionName', str), ('parentName', str),
                                                      ('length', float), ('sectionType', str)))
        self.sp = dict(nx.all_pairs_shortest_path(self.G))

        self.parentID_list, self.sectionID_list, self.sectionName_list, self.length_list = parentID_list, sectionID_list, sectionName_list, length_list
    
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
    def add_synapses(self, numSyn=1000):
        self.numSyn = numSyn
        sectionID_list, sectionName_list = self.sectionID_list, self.sectionName_list
        synapses_list, netstims_list, netcons_list, randoms_list = [], [], [], []
    
        sections_basal = [i for i in map(list, list(self.complex_cell.basal))] 
        sections_apical = [i for i in map(list, list(self.complex_cell.apical))]

        rnd = self.rnd
        if rnd.uniform() < 0.85:
            e_syn, tau1, tau2, spike_interval, syn_weight = 0, 0.3, 1.8, 1000/2.5, 0.0016
        else:
            e_syn, tau1, tau2, spike_interval, syn_weight = -86, 1, 8, 1000/15.0, 0.0008

        sectionID_synapse_list, section_synapse_list, segment_synapse_list = [], [], []
        
        type_array = np.array([''] * numSyn)
        loc_array = np.zeros(numSyn)
        
        for i in tqdm(range(numSyn)):
            Section = rnd.choice(sections_basal)
            section = Section[0].sec
            sectionName = section.psection()['name']
            sectionID_synapse = sectionID_list[sectionName_list.index(sectionName)]
            
            section_synapse_list.append(section)
            sectionID_synapse_list.append(sectionID_synapse)

            # Use to differentiate between input type A and B
            # type_array[i] = rnd.choice(['A', 'B'])

            loc = section(rnd.uniform()).x
            loc_array[i] = loc

            segment_synapse = section(loc)
            segment_synapse_list.append(segment_synapse)
            synapses_list.append(h.Exp2Syn(segment_synapse))

            synapses_list[i].e, synapses_list[i].tau1, synapses_list[i].tau2 = e_syn, tau1, tau2

            netstims_list.append(h.NetStim())
            netstims_list[i].interval, netstims_list[i].number, netstims_list[i].start, netstims_list[i].noise = spike_interval, 10, 100, 1

            randoms_list.append(h.Random())
            randoms_list[i].Random123(i)
            randoms_list[i].negexp(1)
            netstims_list[i].noiseFromRandom(randoms_list[i])

            netcons_list.append(h.NetCon(netstims_list[i], synapses_list[i])) # need to rewrite with an assign function
            netcons_list[i].delay, netcons_list[i].weight[0] = 0, syn_weight

            time.sleep(0.01)
        
        self.type_array, self.loc_array, self.sectionID_synapse_list, self.section_synapse_list, self.segment_synapse_list = type_array, loc_array, sectionID_synapse_list, section_synapse_list, segment_synapse_list
        self.netcons_list = netcons_list
        self.visualize_simulation()

    #cannot use jit for this function either
    # @jit
    def set_synapse_type(self):
        # 参数配置
        rnd = self.rnd
        type_list = ['A', 'B']  # 类型列表
        num_clusters_per_type = (5, 11)  # 每种类型的簇数量范围
        min_points = 10
        max_points = 100
        center_type_prob = 1  # 中心点类型概率

        type_array = np.array([''] * self.numSyn, dtype=str)

        num_types = len(type_list)
        cluster_centers = []

        sectionID_synapse_list = self.sectionID_synapse_list
        distance_matrix = self.distance_matrix

        for i in range(num_types):
            num_clusters = np.random.randint(*num_clusters_per_type)
            cluster_centers_type = np.random.choice(sectionID_synapse_list, num_clusters, replace=False)
            cluster_centers.extend([(center, type_list[i]) for center in cluster_centers_type])

        for centers, ptype in cluster_centers:
            idx = np.where(sectionID_synapse_list == centers)[0]
            type_array[idx] = ptype

        while '' in type_array:
            for centers, ptype in cluster_centers:
                num_points = np.random.randint(min_points, max_points + 1)
                if np.count_nonzero(type_array == '') < self.numSyn / 2:
                    num_points = np.random.randint(min_points, min(max_points, 31))  # 较少空白时，num_points较小
                distances = distance_matrix[:, centers]
                eligible_indices = np.where(type_array == '')[0]
                nearest_indices = eligible_indices[np.argsort(distances[eligible_indices])[:num_points]]
                
                # type_array[nearest_indices] = ptype
                non_center_types = [t for t in type_list if t != ptype]
                chosen_type = rnd.choice([ptype] + non_center_types, p=[center_type_prob] + [(1 - center_type_prob) / (num_types - 1)] * (num_types - 1), size=len(nearest_indices))
                type_array[nearest_indices] = chosen_type

        self.type_array = type_array

    def add_clustered_synapses(self, numSyn=1000, k=10, cluster_radius=2.5):
        # self.initialize_cluster_flag = True

        sectionID_list, sectionName_list = self.sectionID_list, self.sectionName_list
        synapses_list, netstims_list, netcons_list, randoms_list = [], [], [], []
        sections_basal_apical = [i for i in map(list, list(self.complex_cell.basal))] + [i for i in map(list, list(self.complex_cell.apical))]
        all_sections = [i for i in map(list, list(self.complex_cell.soma))] + sections_basal_apical

        rnd = self.rnd
        if rnd.uniform() < 0.85:
            e_syn, tau1, tau2, spike_interval, syn_weight = 0, 0.3, 1.8, 1000/2.5, 0.0016
        else:
            e_syn, tau1, tau2, spike_interval, syn_weight = -86, 1, 8, 1000/15.0, 0.0008

        sectionID_synapse_list, section_synapse_list, segment_synapse_list = [], [], []
        type_cluster_list = []
        type_array = np.array([''] * numSyn)
        loc_array = np.zeros(numSyn)
        self.numSyn, self.k, self.cluster_radius = numSyn, k, cluster_radius
        numSyn = numSyn - k

        section_cluster_list, Section_cluster_list, loc_lower_bound_list, loc_upper_bound_list = [], [], [], []
        
        type_list = ['A', 'B']
        num_types = len(type_list)
        center_type_prob = 1

        for i in tqdm(range(k)):
            
            # could add a new attribute, order, for each section; 
            # and then extract different number of sections from different orders
            Section = rnd.choice(sections_basal_apical)
            section = Section[0].sec

            Section_cluster_list.append(Section)
            section_cluster_list.append(section)

            sectionName = section.psection()['name']
            sectionID_synapse = sectionID_list[sectionName_list.index(sectionName)]
            section_synapse_list.append(section)
            sectionID_synapse_list.append(sectionID_synapse)

            type = rnd.choice(['A', 'B'])
            type_array[i] = type
            type_cluster_list.append(type)

            loc = section(rnd.uniform()).x
            loc_array[i] = loc

            loc_lower_bound = loc - cluster_radius / section.L
            loc_upper_bound = loc + cluster_radius / section.L
            loc_lower_bound_list.append(loc_lower_bound)
            loc_upper_bound_list.append(loc_upper_bound)
            
            segment_synapse = section(loc)
            segment_synapse_list.append(segment_synapse)
            synapses_list.append(h.Exp2Syn(segment_synapse))

            synapses_list[i].e, synapses_list[i].tau1, synapses_list[i].tau2 = e_syn, tau1, tau2

            netstims_list.append(h.NetStim())
            netstims_list[i].interval, netstims_list[i].number, netstims_list[i].start, netstims_list[i].noise = spike_interval, 10, 100, 1

            randoms_list.append(h.Random())
            randoms_list[i].Random123(i)
            randoms_list[i].negexp(1)
            netstims_list[i].noiseFromRandom(randoms_list[i])

            netcons_list.append(h.NetCon(netstims_list[i], synapses_list[i])) # need to rewrite with an assign function
            netcons_list[i].delay, netcons_list[i].weight[0] = 0, syn_weight

        for i in tqdm(range(numSyn)):
            section_cluster = rnd.choice(section_cluster_list)
            Section_cluster = Section_cluster_list[section_cluster_list.index(section_cluster)]
            loc_lower_bound = loc_lower_bound_list[section_cluster_list.index(section_cluster)]
            loc_upper_bound = loc_upper_bound_list[section_cluster_list.index(section_cluster)]
            type_cluster = type_cluster_list[section_cluster_list.index(section_cluster)]

            sectionName_cluster = section_cluster.psection()['name']
            sectionID_synapse_cluster = sectionID_list[sectionName_list.index(sectionName_cluster)]

            loc = rnd.uniform(loc_lower_bound,loc_upper_bound) 
            if loc > 1 or loc < 0:
                gap = floor(loc)
                loc = loc - gap
                # Find the new section with Networkx (successors or predecessors)
                if gap < 0:
                    if list(self.DiG.predecessors(sectionID_synapse_cluster)) != []:
                        sectionID_synapse_cluster = next(list(self.DiG.predecessors(sectionID_synapse_cluster))[0] for _ in range(abs(gap)))
                        # while gap < 0:
                        #     sectionID_synapse_cluster = list(self.DiG.predecessors(sectionID_synapse_cluster))[0]
                        #     gap += 1
                    else:
                        loc, gap = 0, 0
                elif gap > 0:
                    
                    if list(self.DiG.successors(sectionID_synapse_cluster)) != []:
                        sectionID_synapse_cluster = next(list(self.DiG.successors(sectionID_synapse_cluster))[0]for _ in range(gap))
                        # while gap > 0:
                        #     sectionID_synapse_cluster = list(self.DiG.successors(sectionID_synapse_cluster))[0]
                        #     gap -= 1
                    else:
                        loc, gap = 1, 0

                Section_cluster = all_sections[sectionID_synapse_cluster]
                # Section_cluster = sections_basal_apical[sections_basal_apical.index(Section_cluster) + gap]
                section_cluster = Section_cluster[0].sec

            section_synapse_list.append(section_cluster)
            sectionID_synapse_list.append(sectionID_synapse_cluster)
            loc_array[i+k] = loc   

            non_center_types = [t for t in type_list if t != type_cluster]
            type_array[i+k] = rnd.choice([type_cluster] + non_center_types, p=[center_type_prob] + [(1 - center_type_prob) / (num_types - 1)] * (num_types - 1))
            
            segment_synapse = section_cluster(loc)
            segment_synapse_list.append(segment_synapse)
            synapses_list.append(h.Exp2Syn(segment_synapse))

            synapses_list[i+k].e, synapses_list[i+k].tau1, synapses_list[i+k].tau2 = e_syn, tau1, tau2

            netstims_list.append(h.NetStim())
            netstims_list[i+k].interval, netstims_list[i+k].number, netstims_list[i+k].start, netstims_list[i+k].noise = spike_interval, 6, 50, 1

            randoms_list.append(h.Random())
            randoms_list[i+k].Random123(i+k)
            randoms_list[i+k].negexp(1)
            netstims_list[i+k].noiseFromRandom(randoms_list[i+k])

            netcons_list.append(h.NetCon(netstims_list[i+k], synapses_list[i+k])) # need to rewrite with an assign function
            netcons_list[i+k].delay, netcons_list[i+k].weight[0] = 0, syn_weight

            time.sleep(0.01)

        self.loc_array, self.sectionID_synapse_list, self.section_synapse_list, self.segment_synapse_list = loc_array, sectionID_synapse_list, section_synapse_list, segment_synapse_list
        self.type_array = type_array
        # self.visualize_simulation()

        return type_array

    #cannot use jit for this function
    # @jit
    def calculate_distance_matrix(self, distance_limit=2000):
        loc_array, sectionID_synapse_list = self.loc_array, self.sectionID_synapse_list
        parentID_list, length_list = self.parentID_list, self.length_list
        
        length_list = np.array(length_list)
        distance_matrix = np.zeros((self.numSyn, self.numSyn))
        for i in range(self.numSyn):
            for j in range(i + 1, self.numSyn):
                    m = sectionID_synapse_list[i]
                    n = sectionID_synapse_list[j]

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
        self._recursive_plot(s, self.segment_synapse_list)
        plt.title(title)
        # plt.show()
 
    def visualize_simulation(self):
        soma_v = h.Vector().record(self.complex_cell.soma[0](0.5)._ref_v)
        dend_v = h.Vector().record(self.complex_cell.dend[0](0.5)._ref_v)
        apic_v = h.Vector().record(self.complex_cell.apic[0](0.5)._ref_v)
        time_v = h.Vector().record(h._ref_t)
        
        spike_times = [h.Vector() for nc in self.netcons_list]
        for nc, spike_times_vec in zip(self.netcons_list, spike_times):
            nc.record(spike_times_vec)

        h.tstop = 1000
        st = time.time()
        h.run()
        print('complex cell simulation time {:.4f}'.format(time.time()-st))

        # h.finitialize(-65 * mV)
        # h.continuerun(100 * ms)

        # plotting the results

        plt.figure(figsize=(5, 5))
        for i, spike_times_vec in enumerate(spike_times[4:11]):
            plt.vlines(spike_times_vec, i + 0.5, i + 1.5)

        plt.figure(figsize=(5, 5))
        plt.plot(time_v, soma_v, label='soma')
        plt.plot(time_v, dend_v, label='basal')
        plt.plot(time_v, apic_v, label='apical')

        plt.legend()
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        # plt.show()

    def visualize_distance(self):
        type_array, distance_matrix = self.type_array, self.distance_matrix
        
        distance_list, distance_a_a_list, distance_b_b_list = [], [], []
        for i in range(self.numSyn):
            for j in range(self.numSyn):
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

