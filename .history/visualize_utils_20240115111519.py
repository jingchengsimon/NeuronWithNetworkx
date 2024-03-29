from neuron import gui, h
import os
import matplotlib.pyplot as plt

def visualize_synapses(self, folder_path, title='Synapses'):
        s = h.PlotShape(False)
        self._recursive_plot(s, self.section_synapse_df['segment_synapse'].values)
        plt.title(title)
        
        file_path = os.path.join(folder_path, f'figure_synapses.png')
        plt.savefig(file_path)
        plt.close()


def visualize_simulation(self, visualization_params):
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
 