from neuron import h
import os
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly

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
        plt.figure(figsize=(5, 5))
        plt.title(str(ori_dg) + '-' + str(stim_id) + ' clustered spike raster')
        for i, spike_times_vec in enumerate(spike_times_clustered):
            try:
                if len(spike_times_vec) > 0:
                    plt.vlines(spike_times_vec, i + 0.5, i + 1.5)
            except IndexError:
                continue  
        
        plt.show()

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

def visualize_summary_simulation(time_v, dend_v_list, dend_v_peak_list, soma_v_list, soma_v_peak_list, folder_path):
    plt.figure()
    for i in range(0, len(dend_v_list), 4):
        plt.plot(time_v, dend_v_list[-(i+1)], label='dend'+str(len(dend_v_list)-i))
        plt.plot(time_v, soma_v_list[-(i+1)], label='soma'+str(len(soma_v_list)-i))
    plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('Dendritic volatge (mV)')
    file_path = os.path.join(folder_path, f'figure_volatge_time.png')
    plt.savefig(file_path)
    plt.close(i)

    plt.figure()
    plt.plot(range(1,len(dend_v_peak_list)+1), dend_v_peak_list, 'r', label='Dendritic peak volatge')
    plt.plot(range(1,len(soma_v_peak_list)+1), soma_v_peak_list, 'b', label='Soma peak volatge')
    plt.legend()
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

def visualize_morpho(section_synapse_df, soma_v, seg_v):
    # Choose the time point with the maximum soma voltage
    # Find the index that soma_v first cross 0
    max_soma_idx = np.argmax(soma_v[300*40:]) - 4
    print(f"Max soma idx: {max_soma_idx}")

    v_vals = [seg_v[i][max_soma_idx] for i in range(len(seg_v))]

    # Reset the voltage
    seg_list = [seg for sec in h.allsec() for seg in sec]
    for i in range(len(seg_list)):
        seg_list[i].v = v_vals[i]

    ps = h.PlotShape(False) # default variable is voltage
    ps.variable('v')
    # ps.scale(min(v_vals), max(v_vals))
    ps.scale(-70, 0)

    # Create a custom colormap using Matplotlib (cool colormap)
    cmap = cm.cool
    seg_syn_df = section_synapse_df[section_synapse_df['type'] == 'C']['segment_synapse'].values
    soma = section_synapse_df[section_synapse_df['region'] == 'soma']['section_synapse'].values[0]
    fig1=ps.plot(plotly, cmap=cmap).mark(soma(0.5))

    fig2=ps.plot(plotly, cmap=cmap).mark(soma(0.5))
    for seg_syn in seg_syn_df:
        fig2 = fig2.mark(seg_syn)

    # Set the axis limits for x, y, and z to be the same
    axis_limit = [-200, 1200] 
    fig1.update_layout(scene=dict(
        xaxis=dict(range=axis_limit),
        yaxis=dict(range=axis_limit),
        zaxis=dict(range=axis_limit)
    ))

    fig2.update_layout(scene=dict(
        xaxis=dict(range=axis_limit),
        yaxis=dict(range=axis_limit),
        zaxis=dict(range=axis_limit)
    ))

    # Create a colormap function
    colormap = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1)).to_rgba
    plotly_colorscale = [[v, f'rgb{tuple(int(255 * c) for c in colormap(v)[:3])}'] for v in np.linspace(0, 1, cmap.N)]
    colorbar_trace = go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        marker=dict(
            colorscale=plotly_colorscale,
            cmin=-70, #min(v_vals),
            cmax=0, #max(v_vals),
            colorbar=dict(
                title='v (mV)',
                thickness=20  # Adjust the thickness of the colorbar
            ),
            showscale=True
        )
    )

    # Add the colorbar trace to the figure
    # add title
    fig1.update_layout(title_text=f'Morphology t={max_soma_idx}')
    fig1.add_trace(colorbar_trace)
    fig1.update_xaxes(showticklabels=False, showgrid=False)
    fig1.update_yaxes(showticklabels=False, showgrid=False)
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)')

    fig2.update_layout(title_text=f'Morphology with synapses t={max_soma_idx}')
    fig2.add_trace(colorbar_trace)  
    fig2.update_xaxes(showticklabels=False, showgrid=False)
    fig2.update_yaxes(showticklabels=False, showgrid=False)
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)')

    fig1.write_image(f"morpho_{max_soma_idx}.pdf", engine="kaleido")
    fig2.write_image(f"morpho_syn_{max_soma_idx}.pdf", engine="kaleido")

    fig1.show()
    fig2.show()