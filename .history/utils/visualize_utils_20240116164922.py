from neuron import gui, h
import os
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_synapses(section_synapse_df, folder_path, title='Synapses'):
        s = h.PlotShape(False)
        recursive_plot(s, section_synapse_df['segment_synapse'].values, section_synapse_df['type'].values)
        plt.title(title)
        
        file_path = os.path.join(folder_path, f'figure_synapses.png')
        plt.savefig(file_path)
        plt.close()

def recursive_plot(s, seg_list, type_array, index=0):
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
        return recursive_plot(s.plot(plt), seg_list, index+1)
    elif index <= len(seg_list):
        # if self.initialize_cluster_flag == False:
        segment_type = type_array[index - 1]
        marker = markers.get(segment_type, 'or')  # 如果类型不在字典中，默认使用'or'作为标记
        return recursive_plot(s.mark(seg_list[index - 1], marker), seg_list, index + 1)
    
            # if self.type_array[index-1] == 'A':
            #     return self._recursive_plot(s.mark(seg_list[index-1],'or'), seg_list, index+1)
            # else:
            #     return self._recursive_plot(s.mark(seg_list[index-1],'xb'), seg_list, index+1)
        # else:
        #     return self._recursive_plot(s.mark(seg_list[index-1],'xr'), seg_list, index+1)
        
def visualize_simulation(visualization_params, num_syn_to_get_input, folder_path):
        ori_dg, stim_id, soma_v, dend_v, apic_v, time_v, spike_times_clustered, dend_i, dend_i_nmda, dend_i_ampa, spike_times = visualization_params
        
        # plotting the results
        # plt.figure(figsize=(5, 5))
        # plt.title(str(ori_dg) + '-' + str(stim_id) + ' clustered spike raster')
        # for i, spike_times_vec in enumerate(spike_times_clustered):
        #     try:
        #         if len(spike_times_vec) > 0:
        #             plt.vlines(spike_times_vec, i + 0.5, i + 1.5)
        #     except IndexError:
        #         continue  

        # plt.figure(figsize=(5, 5))
        # plt.title(str(ori_dg) + '-' + str(stim_id)+ ' all spike raster')
        # for i, spike_times_vec in enumerate(spike_times):
        #     try:
        #         if len(spike_times_vec) > 0:
        #             plt.vlines(spike_times_vec, i + 0.5, i + 1.5)
        #     except IndexError:
        #         continue


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
        # plt.plot(time_v, apic_v, label='apical')
        plt.plot(time_v, dend_v, label='basal')
        plt.plot(time_v, soma_v, label='soma')
        plt.legend()
        plt.ylim(-90, 40)
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        plt.title(str(ori_dg) + '-' + str(stim_id))

        plt.figure(figsize=(5, 5))
        plt.plot(time_v, dend_i, label='dendritic current')
        plt.plot(time_v, dend_i_nmda, label='dendritic current nmda')
        plt.plot(time_v, dend_i_ampa, label='dendritic current ampa')
        plt.legend()
        plt.xlabel('Time (ms)')
        plt.ylabel('Current (nA)')
        plt.title(str(ori_dg) + '-' + str(stim_id)+' dendritic current')

        plt.figure(figsize=(5, 5))
        plt.plot(time_v, dend_i_nmda, label='dendritic current nmda')
        plt.legend()
        plt.xlabel('Time (ms)')
        plt.ylabel('Current (nA)')
        plt.title(str(ori_dg) + '-' + str(stim_id)+' dendritic current nmda')

        plt.figure(figsize=(5, 5))
        plt.plot(time_v, dend_i_ampa, label='dendritic current ampa')
        plt.legend()
        plt.xlabel('Time (ms)')
        plt.ylabel('Current (nA)')
        plt.title(str(ori_dg) + '-' + str(stim_id)+' dendritic current ampa')

        for i in plt.get_fignums():
            plt.figure(i)
            file_path = os.path.join(folder_path, f'figure_{ori_dg}_{stim_id}_{num_syn_to_get_input}_{i}.png')
            plt.savefig(file_path)
            plt.close(i)

def visualize_summary_simulation(time_v, dend_v_list, dend_v_peak_list, folder_path):
    plt.figure()
    for i in range(0, len(dend_v_list), 4):
        plt.plot(time_v, dend_v_list[-(i+1)], label=len(dend_v_list)-i)
    plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('Dendritic volatge (mV)')
    file_path = os.path.join(folder_path, f'figure_volatge_time.png')
    plt.savefig(file_path)
    plt.close(i)

    plt.figure()
    plt.plot(dend_v_peak_list)
    plt.xlabel('Number of synapses')
    plt.ylabel('Peak Dendritic volatge (mV)')
    file_path = os.path.join(folder_path, f'figure_volatge_numOfSyn.png')
    plt.savefig(file_path)
    plt.close(i)

     
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
