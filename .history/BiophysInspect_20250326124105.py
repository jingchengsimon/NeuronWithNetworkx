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
    
def nonlinearity_visualization(exp, k, fig, ax, data_type, stack_flag, exp_idx, alpha=1):
    try:
        v, i, nmda, ampa, nmda_g, ampa_g, soma, apic_v, apic_ica, soma_i, \
        trunk_v, basal_v, tuft_v, basal_bg_i_nmda, basal_bg_i_ampa, \
        tuft_bg_i_nmda, tuft_bg_i_ampa, dt, simu_info, sec_syn_df = load_data(exp)
    
    except ValueError:
        v, i, nmda, ampa, nmda_g, ampa_g, soma, apic_v, apic_ica, dt, simu_info = load_data(exp)
    
    if v.ndim == 5:
        v = np.mean(v, axis=2) # shape: [num_clusters, num_times, num_affs, num_trials]
        i = np.mean(i, axis=2) # shape: [num_clusters, num_times, num_affs, num_trials]
        nmda = np.mean(nmda, axis=2)
        ampa = np.mean(ampa, axis=2)
        nmda_g = np.mean(nmda_g, axis=2)
        ampa_g = np.mean(ampa_g, axis=2)
        
        soma = np.mean(soma, axis=1) # shape: [num_times, num_affs, num_trials]
        apic_v = np.mean(apic_v, axis=1) # shape: [num_times, num_affs, num_trials]

    t = simu_info['time point of stimulation']
    t_start, t_end = (t-20)*40, (t+100)*40
    syn_num = v.shape[2]
    syn_num_step = 1
    # Check if soma fires an AP between -20 to 20 ms
    soma_AP_flag = np.max(soma[(t-20)*40:(t+20)*40, :, :]) > 0

    fig.subplots_adjust(wspace=0)

    num_clus_sampled = np.min([v.shape[0], 6])
    color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    
    ax.flat[k//syn_num_step].set_title(f'{exp_idx+1}') # Label the subplot with the epoch index
    
    scale_factor_dend, scale_factor_root = 1, 1 # Scale the traces for better visualization
    
    v_baseline = np.average(v[:, (t-200)*40:(t-100)*40, :, :])
    soma_baseline = np.average(soma[(t-200)*40:(t-100)*40, :, :])
    apic_v_baseline = np.average(apic_v[(t-200)*40:(t-100)*40, :, :])

    num_preunit = 72
    syn_num_list = list(range(0, num_preunit + 2, 2))
    
    if data_type == 'dend':
        ax.flat[k//syn_num_step].set_ylim(-3, 80)
        ax.flat[k//syn_num_step].set_yticks(list(range(0, 80, 10)))

    ax.flat[k//syn_num_step].set_ylabel('EPSP (mV)')
    ax.flat[k//syn_num_step].set_xlabel('Number of Synapses')
    ax.flat[k//syn_num_step].set_xticks(list(range(0, num_preunit + 12, 12)))
    
    if data_type == 'dend':
        for clus_idx in range(num_clus_sampled):
            EPSP_list = scale_factor_dend*np.max(np.average((v[clus_idx, t_start:t_end, :, :]-v_baseline),axis=-1),axis=0)
            ax.flat[k//syn_num_step].plot(syn_num_list, EPSP_list, color=color_list[clus_idx], alpha=1)
            # add expected linear fit (EPSP_list[1] - EPSP_list[0]) * syn_num_list + EPSP_list[0]
            # ax.flat[k//syn_num_step].plot(syn_num_list, (EPSP_list[1] - EPSP_list[0]) * np.array(syn_num_list) + EPSP_list[0], color=color_list[clus_idx], linestyle='--', alpha=0.5)
    elif (data_type == 'nexus') and ('basal' in exp):
            soma_EPSP_list = scale_factor_root*np.max(np.average((soma[t_start:t_end, :, :]-soma_baseline),axis=-1),axis=0)
            ax.flat[k//syn_num_step].plot(syn_num_list, soma_EPSP_list, color='k', alpha=1)
            # ax.flat[k//syn_num_step].plot(syn_num_list, (soma_EPSP_list[1] - soma_EPSP_list[0]) * np.array(syn_num_list) + soma_EPSP_list[0], color='k', linestyle='--', alpha=0.5)
    elif (data_type == 'nexus') and ('apical' in exp):
            apic_EPSP_list = scale_factor_root*np.max(np.average((apic_v[t_start:t_end, :, :]-apic_v_baseline),axis=-1),axis=0)
            ax.flat[k//syn_num_step].plot(syn_num_list, apic_EPSP_list, color='b', alpha=1)
            # ax.flat[k//syn_num_step].plot(syn_num_list, (apic_EPSP_list[1] - apic_EPSP_list[0]) * np.array(syn_num_list) + apic_EPSP_list[0], color='b', linestyle='--', alpha=0.5)
            
    g_NMDA = simu_info['initial weight of AMPANMDA synapses'] * 1000
    return v, soma, apic_v, g_NMDA

def full_nonlinearity_visualization(exp_list, idx_list, data_list, stack_flag=False, num_epochs=10):
    t = 500
    num_figs = num_epochs if not stack_flag else 1
    
    figs, axs = [], []

    num_ax_rows = 3
    for i in range(num_figs):
        fig, ax = plt.subplots(num_ax_rows, num_epochs//num_ax_rows, figsize=((num_epochs) * 3//num_ax_rows, 4*num_ax_rows), sharey=True)
        # fig, ax = plt.subplots(1, 10, figsize=((10) * 2, 4), sharey=True)
        plt.suptitle(exp_list[0] + ' ' + str(idx_list[0]) + ' ' + data_list[0], fontsize=18)
        figs.append(fig)
        axs.append(ax)
    
    dend_trace_list, soma_trace_list, apic_trace_list = [], [], []
    
    for epoch_idx in range(num_epochs):
        # if not stack_flag:
        #     v, soma, apic_v, g_NMDA = nonlinearity_visualization(exp_list[0] + '/' + str(idx_list[0]) + '/' + str(exp_idx + 1) + '/', 
        #                         exp_idx, figs[exp_idx], axs[exp_idx], data_list[0], stack_flag, exp_idx)
        # else:
        dend, soma, apic_v, g_NMDA = nonlinearity_visualization(exp_list[0] + '/' + str(idx_list[0]) + '/' + str(epoch_idx + 1) + '/', 
                            epoch_idx, figs[0], axs[0], data_list[0], stack_flag, epoch_idx)

        dend_trace_list.append(dend)
        soma_trace_list.append(soma)
        apic_trace_list.append(apic_v)
        
    ## Sort the axes based on Soma EPSP amplitude and replot the figures
    if data_list[0] == 'dend':
        max_values = [np.max(dend_trace[:, (t-20)*40:(t+100)*40, :, :]) for dend_trace in dend_trace_list]
    elif (data_list[0] == 'nexus') & ('basal' in exp_list[0]):
        max_values = [np.max(soma_trace[(t-20)*40:(t+100)*40, :, :]) for soma_trace in soma_trace_list]
    elif (data_list[0] == 'nexus') & ('apical' in exp_list[0]):
        max_values = [np.max(apic_trace[(t-20)*40:(t+100)*40, :, :]) for apic_trace in apic_trace_list]

    sorted_indices = np.argsort(max_values)[::-1]

    print(f'{anal_loc} range{range_idx} {data_list[0]} gNMDA{g_NMDA}: ', list(sorted_indices+1))
    
    figs, axs = [], []

    for i in range(num_figs):
        fig, ax = plt.subplots(num_ax_rows, num_epochs//num_ax_rows, figsize=((num_epochs) * 3//num_ax_rows, 4*num_ax_rows), sharey=True)
        plt.suptitle(exp_list[i] + ' ' + str(idx_list[i]) + ' ' + data_list[i], fontsize=18)
        figs.append(fig)
        axs.append(ax)
    
    for epoch_idx in range(num_epochs):
        sorted_epoch_idx = sorted_indices[epoch_idx] 
        v, soma, apic_v, g_NMDA = nonlinearity_visualization(exp_list[0] + '/' + str(idx_list[0]) + '/' + str(sorted_epoch_idx + 1) + '/', 
                                epoch_idx, figs[0], axs[0], data_list[0], stack_flag, sorted_epoch_idx)

    for fig in figs:
        fig.tight_layout()

    plt.savefig(f'/G/results/simulation/202503_Nonlinearity/{anal_loc}_range{range_idx}_{data_list[0]}_gNMDA{g_NMDA}_{spat_cond}_invitro_fixedW_tau90_addNaK.jpg')
    
# create path
if not os.path.exists('/G/results/simulation/202503_Nonlinearity/'):
    os.makedirs('/G/results/simulation/202503_Nonlinearity/')

anal_loc, range_idx, col_idx, num_epochs = 'basal', 0, 0, 30 # range_idx: 0-2, col_idx: 0-5
iter_start_idx, iter_end_idx = 1+(col_idx)*7, 1+(col_idx)*7+1
iter_step, iter_times = 1, iter_end_idx - iter_start_idx


for folder_idx in range(1,2):
    iter_start_idx, iter_end_idx = 1+(col_idx)*7+folder_idx, 1+(col_idx)*7+1+folder_idx
    for anal_loc in ['basal', 'apical']:
        for spat_cond in ['clus']:
            for range_idx in range(3):
                for attr in ['dend', 'nexus']:
                    full_nonlinearity_visualization([anal_loc+f'_range{range_idx}_{spat_cond}_invio_fixedW_tau90_addNaK_overallbranchtest_varynumsyn_step2'] * iter_times,
                                                list(range(iter_start_idx, iter_end_idx, iter_step)),
                                                [attr] * iter_times,
                                                stack_flag=True, num_epochs=num_epochs)