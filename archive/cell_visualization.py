from cell_with_networkx import CellWithNetworkx

class CellVisualization(CellWithNetworkx):
    def __init__(self, swc_file_path):
        super().__init__(swc_file_path)
        self.cell = CellWithNetworkx(swc_file_path)

    def visualize_synapses(self, folder_path, title='Synapses'):
        s = h.PlotShape(False)
        self._recursive_plot(s, self.section_synapse_df['segment_synapse'].values)
        plt.title(title)
        
        file_path = os.path.join(folder_path, f'figure_synapses.pdf')
        plt.savefig(file_path)
        plt.close()
    
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

        for spike_count_list in spike_counts:
            print('num_spikes across orientation: ', spike_count_list)
            print('OSI of the model neuron: ', self.calculate_osi(spike_count_list, ori_dg_list, pref_ori_dg))
            print('cirvar: ', self.calculate_cirvar(spike_count_list, ori_dg_list))

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

    def _visualize_simulation(self, visualization_params):
        ori_dg, stim_id, stim_index, soma_v, dend_v, apic_v, time_v, spike_times_clustered = visualization_params
        
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

        # threshold to define spikes
        threshold = 0

        num_spikes = self._count_spikes(soma_v, threshold)
        print(str(ori_dg)+'-'+str(stim_id))
        print("Number of spikes:", num_spikes)
        
        self.num_spikes_df.at[ori_dg, stim_index] = num_spikes
        # self.num_spikes_list.append(num_spikes)