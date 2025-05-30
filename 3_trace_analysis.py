import os
import json
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
from scipy.signal import find_peaks, butter, filtfilt
from scipy.optimize import curve_fit
from scipy.ndimage import binary_opening, label
import seaborn as sns
import ast
from matplotlib.animation import FuncAnimation

# ignore runtime warning
warnings.filterwarnings("ignore", category=RuntimeWarning)

root_folder_path = '/G/results/simulation/'

def load_data(exp):
    
    try:
        v_path = [root_folder_path, exp, 'dend_v_array.npy']
        i_path = [root_folder_path, exp, 'dend_i_array.npy']
        nmda_path = [root_folder_path, exp, 'dend_nmda_i_array.npy']
        ampa_path = [root_folder_path, exp, 'dend_ampa_i_array.npy']
        
        nmda_g_path = [root_folder_path, exp, 'dend_nmda_g_array.npy']
        ampa_g_path = [root_folder_path, exp, 'dend_ampa_g_array.npy']
        
        soma_path = [root_folder_path, exp, 'soma_v_array.npy']
        apic_v_path = [root_folder_path, exp, 'apic_v_array.npy']
        apic_ica_path = [root_folder_path, exp, 'apic_ica_array.npy']

        soma_i_path = [root_folder_path, exp, 'soma_i_array.npy']
            
        trunk_v_path = [root_folder_path, exp, 'trunk_v_array.npy']
        basal_v_path = [root_folder_path, exp, 'basal_v_array.npy']
        tuft_v_path = [root_folder_path, exp, 'tuft_v_array.npy']

        basal_bg_i_nmda_path = [root_folder_path, exp, 'basal_bg_i_nmda_array.npy']
        basal_bg_i_ampa_path = [root_folder_path, exp, 'basal_bg_i_ampa_array.npy']
        tuft_bg_i_nmda_path = [root_folder_path, exp, 'tuft_bg_i_nmda_array.npy']
        tuft_bg_i_ampa_path = [root_folder_path, exp, 'tuft_bg_i_ampa_array.npy']

        v = np.load(os.path.join(*v_path))
        i = np.load(os.path.join(*i_path))
        nmda = np.load(os.path.join(*nmda_path))
        ampa = np.load(os.path.join(*ampa_path))
        
        nmda_g = np.load(os.path.join(*nmda_g_path))
        ampa_g = np.load(os.path.join(*ampa_g_path))
        
        soma = np.load(os.path.join(*soma_path))
        apic_v = np.load(os.path.join(*apic_v_path))
        apic_ica = np.load(os.path.join(*apic_ica_path))

        soma_i = np.load(os.path.join(*soma_i_path))

        trunk_v = np.load(os.path.join(*trunk_v_path))
        basal_v = np.load(os.path.join(*basal_v_path))
        tuft_v = np.load(os.path.join(*tuft_v_path))

        basal_bg_i_nmda = np.load(os.path.join(*basal_bg_i_nmda_path))
        basal_bg_i_ampa = np.load(os.path.join(*basal_bg_i_ampa_path))
        tuft_bg_i_nmda = np.load(os.path.join(*tuft_bg_i_nmda_path))
        tuft_bg_i_ampa = np.load(os.path.join(*tuft_bg_i_ampa_path)) 

    except FileNotFoundError:
        pass

    dt = 1/40000

    # read info from json and add to plot
    with open(os.path.join(root_folder_path, exp, 'simulation_params.json')) as f:
        simu_info = json.load(f)
    
    with open(os.path.join(root_folder_path, exp, 'section_synapse_df.csv')) as f:
        sec_syn_df = pd.read_csv(f)

    try:
        return v, i, nmda, ampa, nmda_g, ampa_g, soma, apic_v, apic_ica, soma_i, \
               trunk_v, basal_v, tuft_v, basal_bg_i_nmda, basal_bg_i_ampa, \
               tuft_bg_i_nmda, tuft_bg_i_ampa, dt, simu_info, sec_syn_df
    
    except NameError:
        return simu_info, sec_syn_df

def visualization(exp, syn_num=0, trial_idx=0, t_start=0, t_end=1000, spk_t_end=1000):
    try:
        v, i, nmda, ampa, nmda_g, ampa_g, soma, apic_v, apic_ica, soma_i, \
        trunk_v, basal_v, tuft_v, basal_bg_i_nmda, basal_bg_i_ampa, \
        tuft_bg_i_nmda, tuft_bg_i_ampa, dt, simu_info, sec_syn_df = load_data(exp)
    
    except ValueError:
        v, i, nmda, ampa, nmda_g, ampa_g, soma, apic_v, apic_ica, dt, simu_info = load_data(exp)
    
    if v.ndim == 5:
        v = np.mean(v, axis=2) # shape: [num_clusters, num_times, num_affs, num_trials]
        soma = np.mean(soma, axis=1) # shape: [num_times, num_affs, num_trials]
        apic_v = np.mean(apic_v, axis=1) # shape: [num_times, num_affs, num_trials]
        trunk_v = np.mean(trunk_v, axis=1) # shape: [num_times, num_affs, num_trials]
        basal_v = np.mean(basal_v, axis=1) # shape: [num_times, num_affs, num_trials]
        tuft_v = np.mean(tuft_v, axis=1) # shape: [num_times, num_affs, num_trials]
        
    t = simu_info['time point of stimulation']
    t_start, t_end = t_start*40, t_end*40
    t_vals = np.arange(t_start, t_end)*dt 
    
    color_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
    num_clus = np.min([v.shape[0], len(color_list)])
    
    # Set up the figure
    num_subplots = 7
    fig1, axes = plt.subplots(num_subplots, 1, figsize=(40*(t_end-t_start)//(spk_t_end*40), num_subplots*1.5), sharex=True)
    plt.ion()

    # Main title
    axes[0].set_title('Membrane potentials')# recorded simultaneously across the dendritic tree and soma during naturalistic drive')
    axes[0].set_ylim(-75, 40)
    axes[1].set_ylim(-75, 20)
    axes[2].set_ylim(-80, 40)
    
    axes[3].set_ylim(-75, 40)
    # axes[4].set_ylim(-80, -10)
    # axes[5].set_ylim(-80, -10)

    # for syn_num in [0,5,15,20,30,36]:
    for syn_num in range(0, v.shape[2], 1):
        alpha = min(1, 0.2+0.8*(syn_num+1)/v.shape[2])
        # Subplot 1: Soma; Subplot 2: Nexus; Subplot 3: Dend
        axes[0].plot(1000 * t_vals, soma[t_start:t_end, syn_num, trial_idx].squeeze(), alpha=alpha, color='k')#, label='Soma EPSP')
        axes[1].plot(1000 * t_vals, apic_v[t_start:t_end, syn_num, trial_idx].squeeze(), alpha=alpha, color='salmon')#, label='Calcium Zone')
        
        for clus_idx in range(num_clus):
            alpha_dend = alpha if num_clus == 1 else 0.3
            axes[2].plot(1000 * t_vals, v[clus_idx, t_start:t_end, syn_num, trial_idx].squeeze(), alpha=alpha_dend, color=color_list[clus_idx])#, label='Dend Voltage')
            # axes[2].plot(1000 * t_vals, tuft_v[t_start:t_end, syn_num, trial_idx].squeeze(), alpha=0.9, color='purple', label='Tuft Voltage')
        
        # Subplot 4: Trunk; Subplot 5: Basal; Subplot 6: Tuft
        axes[3].plot(1000 * t_vals, trunk_v[t_start:t_end, syn_num, trial_idx].squeeze(), alpha=alpha, color='orange')#, label='Trunk Voltage')
        # axes[4].plot(1000 * t_vals, basal_v[t_start:t_end, syn_num, trial_idx].squeeze(), alpha=alpha, color='navy')#, label='Basal Voltage')
        # axes[5].plot(1000 * t_vals, tuft_v[t_start:t_end, syn_num, trial_idx].squeeze(), alpha=alpha, color='darkred')#, label='Tuft Voltage')  

    axes[0].plot(1000 * t_vals, np.mean(soma[t_start:t_end, 0, :]) * np.ones_like(t_vals), color='k', linestyle='--', label='Soma Threshold')
    axes[1].plot(1000 * t_vals, np.mean(apic_v[t_start:t_end, 0, :]) * np.ones_like(t_vals), color='salmon', linestyle='--', label='Calcium Threshold')

    # axes[2].set_ylim(-0.01, 0.163)
    axes[2].plot(1000 * t_vals, np.mean(v[:, t_start:t_end, 0, :]) * np.ones_like(t_vals), color=color_list[clus_idx], linestyle='--', label='Dend Threshold')
    # axes[2].plot(1000 * t_vals, np.mean(tuft_v[t_start:t_end, 0, :]) * np.ones_like(t_vals), color='purple', linestyle='--', label='Tuft Threshold')

    axes[3].plot(1000 * t_vals, np.mean(trunk_v[t_start:t_end, 0, :]) * np.ones_like(t_vals), color='orange', linestyle='--', label='Trunk Threshold')
    # axes[4].plot(1000 * t_vals, np.mean(basal_v[t_start:t_end, 0, :]) * np.ones_like(t_vals), color='navy', linestyle='--', label='Basal Threshold')
    # axes[5].plot(1000 * t_vals, np.mean(tuft_v[t_start:t_end, 0, :]) * np.ones_like(t_vals), color='darkred', linestyle='--', label='Tuft Threshold')

    # Set legend should be at the last
    for ax_idx in range(num_subplots-3):
        # axes[ax_idx].set_ylim(-90, 10)    
        axes[ax_idx].set_ylabel('Voltage (mV)')
        axes[ax_idx].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  
        axes[ax_idx].legend(loc='upper right', fontsize=6)
    
    axes[-1].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    axes[-1].set_xlabel('Time (ms)')

    # Create the aster plot (cluster, background excitatory and inhibitory)
    clus_sec_syn_df = sec_syn_df[sec_syn_df['cluster_flag'] == 1]
    exc_bg_sec_syn_df = sec_syn_df[sec_syn_df['type'].isin(['A'])]
    inh_bg_sec_syn_df = sec_syn_df[sec_syn_df['type'].isin(['B'])]
    
    for i, (spike_train, syn_region) in enumerate(zip(clus_sec_syn_df['spike_train'], clus_sec_syn_df['region'])):
        # for syn_num in range(0, v.shape[2], 1):
        try:
            spike_train = ast.literal_eval(spike_train)[-1]  # Convert string to list   
        except ValueError:
            spike_train = []

        if len(spike_train) > 0:
            color = 'purple'
            axes[4].vlines(spike_train, i + 0.5, i + 1.5, color=color, linewidth=6) #2+(syn_num+1)/v.shape[2]*4)  # Set color based on type

    for i, (spike_train, syn_region) in enumerate(zip(exc_bg_sec_syn_df['spike_train_bg'], exc_bg_sec_syn_df['region'])):
        try:
            spike_train = ast.literal_eval(spike_train)[0]  # Convert string to list
        except IndexError:
            spike_train = []
            
        if len(spike_train) > 0:
            color = 'blue' if syn_region == 'basal' else 'red' if syn_region == 'apical' else 'black'
            axes[5].vlines(spike_train, i + 0.5, i + 1.5, color=color, linewidth=6)  # Set color based on type

    for i, (spike_train, syn_region) in enumerate(zip(inh_bg_sec_syn_df['spike_train_bg'], inh_bg_sec_syn_df['region'])):
        # for syn_num in range(0, v.shape[2], 1):
        try:
            spike_train = ast.literal_eval(spike_train)[-1]  # Convert string to list
        except IndexError:
            spike_train = []
            
        if len(spike_train) > 0:
            color = 'blue' if syn_region == 'basal' else 'red' if syn_region == 'apical' else 'black'
            axes[6].vlines(spike_train, i + 0.5, i + 1.5, color=color, linewidth=6) #2+(syn_num+1)/v.shape[2]*4)  # Set color based on type


    # for ax, title in zip(axes[4:], 
    #                      ['Raster Plot of Background Excitatory Spike Trains', 
    #                       'Raster Plot of Background Inhibitory Spike Trains']):
    for ax, title in zip(axes[4:], 
                         ['Raster Plot of Synchronous Excitatory Spike Trains', 
                          'Raster Plot of Background Excitatory Spike Trains', 
                          'Raster Plot of Background Inhibitory Spike Trains']):
        ax.set_xlim(t_start // 40, t_end // 40)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron Index')
        ax.set_title(title)

    # Draw a red dash line at t-20 and t+20 ms
    for ax_idx in range(num_subplots):
        axes[ax_idx].axvline(x=t-20, color='r', linestyle='--', alpha=0.5)
        axes[ax_idx].axvline(x=t, color='r', linestyle='--', alpha=1)
        axes[ax_idx].axvline(x=t+20, color='r', linestyle='--', alpha=0.5)
    
    # Adjust layout to prevent overlap
    fig1.tight_layout() 
    
    # save the figure
    res_path = os.path.join(root_folder_path, '202505_Trace')
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    plt.savefig(os.path.join(res_path, f'mono_simu_{int(t_end/(40*1000))}s_1.png'), dpi=300)
    # return max_EPSC, max_i_list

syn_num, trial_idx = 0, 0
visualization(f'basal_range0_clus_invivo_variedW_tau43_addNaK_monoconn_1s_withAP+Ca_aligned/1/1', syn_num=syn_num, trial_idx=trial_idx, t_start=10, t_end=5000, spk_t_end=5000)