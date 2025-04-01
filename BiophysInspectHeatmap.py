import os
import json
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
import warnings
from scipy.signal import find_peaks, butter, filtfilt
from scipy.optimize import curve_fit
from scipy.ndimage import binary_opening, label
import seaborn as sns

# ignore runtime warning
warnings.filterwarnings("ignore", category=RuntimeWarning)

root_folder_path = '/G/results/simulation/'
  
def find_v_array(folder_path, data_type):
    if data_type == 'Soma':
        return np.load(os.path.join(folder_path, 'soma_v_array.npy'))
    elif data_type == 'Nexus':
        return np.load(os.path.join(folder_path, 'apic_v_array.npy'))
    elif data_type == 'Dend':
        return np.load(os.path.join(folder_path, 'dend_v_array.npy'))

def calculate_mag_values(folder_path, num_epochs=1, data_type='', attr_type='Peak'):
    t = 500
    dt = 1/40000
    t_start, t_end = (t-20)*40, (t+100)*40
    x = np.arange(0, t_end-t_start)*dt

    mag_values = []
    root_v_list = []

    for epoch in range(1, num_epochs+1):
        epoch_folder_path = os.path.join(folder_path, f'{epoch}')
        
        if data_type == 'Soma':
            try:
                soma = find_v_array(epoch_folder_path, data_type)
                if soma.ndim == 4:
                    soma = np.mean(soma, axis=1)
                    soma_baseline = np.mean(soma[(t-200)*40:(t-100)*40, :, :]) # shape: [num_times, num_affs, num_trials]
                if attr_type == 'Peak':
                    # peak
                    peak_soma_spikes = np.max(soma[t_start:t_end,:,:])-np.min(soma[t_start:t_end,:,:])
                    mag_values.append(peak_soma_spikes) 
                elif attr_type == 'Area':
                    # area
                    soma_over_baseline = np.clip(soma[t_start:t_end, :, :] - soma_baseline, 0, None)
                    area_soma_spikes = np.mean(np.trapz(soma_over_baseline, x, axis=0)) # np.mean(np.trapz(calcium_positive, x, axis=0), axis=1)
                    mag_values.append(area_soma_spikes)
                # Average soma voltage trace
                root_v_list.append(np.average(soma[t_start:t_end, 0, :],axis=1).squeeze()-soma_baseline)
                
            except FileNotFoundError:
                pass

        elif data_type == 'Nexus':
            try:
                apic_v = find_v_array(epoch_folder_path, data_type)
                if apic_v.ndim == 4:
                    apic_v = np.mean(apic_v, axis=1)
                    nexus_baseline = np.mean(apic_v[(t-200)*40:(t-100)*40, :, :]) # shape: [num_times, num_affs, num_trials]
                if attr_type == 'Peak':
                    # peak
                    peak_nexus_spikes = np.max(apic_v[t_start:t_end,:,:])-np.min(apic_v[t_start:t_end,:,:])
                    mag_values.append(peak_nexus_spikes) 
                elif attr_type == 'Area':
                    # area
                    nexus_over_baseline = np.clip(apic_v[t_start:t_end, :, :] - nexus_baseline, 0, None)
                    area_nexus_spikes = np.mean(np.trapz(nexus_over_baseline, x, axis=0)) # np.mean(np.trapz(calcium_positive, x, axis=0), axis=1)
                    mag_values.append(area_nexus_spikes)
                # Average apical nexus voltage trace
                root_v_list.append(np.average(apic_v[t_start:t_end, 0, :],axis=1).squeeze()-nexus_baseline)

            except FileNotFoundError:
                pass

        elif data_type == 'Dend':
            try:
                dend_v = find_v_array(epoch_folder_path, data_type)
                if dend_v.ndim == 5:
                    dend_v = np.mean(dend_v, axis=2)
                    dend_baseline = np.mean(dend_v[:,(t-200)*40:(t-100)*40,:,:])
                if attr_type == 'Peak':
                    mag_values.append(np.max(dend_v[:,t_start:t_end,:,:])-np.min(dend_v[:,t_start:t_end,:,:]))
                elif attr_type == 'Area':
                    dend_over_baseline = np.clip(np.mean(dend_v[:,t_start:t_end, :, :], axis=3) - dend_baseline, 0, None)
                    mag_values.append(np.mean(np.trapz(dend_over_baseline, x, axis=1)))
                # Average dend voltage trace
                root_v_list.append(np.average(dend_v[:, t_start:t_end, 0, :], axis=(0, 2)).squeeze()-dend_baseline)
                
            except FileNotFoundError:
                pass
    
    mean_mag_val = np.mean(mag_values)
    mean_root_v = np.mean(root_v_list, axis=0)
    std_root_v = np.std(root_v_list, axis=0)
    
    return mean_mag_val, mean_root_v, std_root_v
    
def create_matrix(root_folder_path, folder_path, num_syn_per_clus, num_clus, num_epochs=1, data_type='', attr_type='Peak'):
    
    mean_mag_val_mat = np.zeros((len(num_syn_per_clus), len(num_clus)))
    # Use dtype=object to store ndarrays
    mean_root_v_mat = np.zeros((len(num_syn_per_clus), len(num_clus)), dtype=object)  
    std_root_v_mat = np.zeros((len(num_syn_per_clus), len(num_clus)), dtype=object)  

    for subfolder_name in os.listdir(os.path.join(root_folder_path, folder_path)):
        subfolder_path = os.path.join(root_folder_path, folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            # Read the simulation_params.json file from the 1st epoch folder
            params_file_path = os.path.join(subfolder_path, '1', 'simulation_params.json')
            with open(params_file_path, 'r') as f:
                params = json.load(f)
            syn_per_clus = params['number of synapses per cluster']
            clus = params['number of clusters']

            i = num_syn_per_clus.index(syn_per_clus)
            j = num_clus.index(clus)

            mean_mag_val, mean_root_v, std_root_v = calculate_mag_values(subfolder_path, num_epochs, data_type, attr_type)
        
            mean_mag_val_mat[i, j] = mean_mag_val
            mean_root_v_mat[i, j] = mean_root_v
            std_root_v_mat[i, j] = std_root_v

    print( "{:.2e}".format(np.min(mean_mag_val_mat)), 
           "{:.2e}".format(np.max(mean_mag_val_mat)))

    return mean_mag_val_mat, mean_root_v_mat, std_root_v_mat

def analyze_case(root_folder_path, folder_path, num_syn_per_clus, num_clus, num_epochs=1, case_name='', data_type='',attr_type='Peak'):
    mag_val_mat, root_v_mat, std_root_v_mat = create_matrix(root_folder_path, folder_path, num_syn_per_clus, num_clus, num_epochs, data_type, attr_type)
    return mag_val_mat, root_v_mat, std_root_v_mat

def calculate_ratio_and_draw_heatmap(mag_val_mat1, mag_val_mat2, 
                                     root_v_mat1, root_v_mat2, 
                                     std_root_v_mat1, std_root_v_mat2,
                                     title, attr1, attr2):
    
    res_path = os.path.join(root_folder_path, '202503_Heatmap')
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    num_rows, num_cols = root_v_mat1.shape
    _, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*1.2, num_rows*1.4)) 

    if 'NMDA vs non-NMDA' in title: 
        plt.suptitle('Ratio of NMDA to non-NMDA ' + title, fontsize=16)
    else:
        plt.suptitle('Ratio of Clustered to Dispersed ' + title, fontsize=16)

    t, dt = 500, 1/40000
    t_start, t_end = (t-20)*40, (t+100)*40
    t_vals = np.arange(t_start, t_end)*dt-t/1000
    
    # Iterate over each subplot and plot the corresponding avg_root_v array
    for i in range(num_rows):
        for j in range(num_cols):
            # Flip the row index to plot [1,1] at the lower-left and [num_rows, num_cols] at the upper-right
            ax = axes[num_rows-1-i, j]
            # ax = axes[i, j]
            root_v1 = root_v_mat1[i, j]
            root_v2 = root_v_mat2[i, j]

            std_root_v1 = std_root_v_mat1[i, j]
            std_root_v2 = std_root_v_mat2[i, j]

            scaled_root_v1, scaled_root_v2, \
            scaled_std_root_v1, scaled_std_root_v2 = root_v1, root_v2, \
                                                     std_root_v1, std_root_v2
            
            ax.plot(t_vals, scaled_root_v1, color='r') # clustered
            ax.fill_between(t_vals, scaled_root_v1 - scaled_std_root_v1, scaled_root_v1 + scaled_std_root_v1, color='r', alpha=0.3)
            ax.plot(t_vals, scaled_root_v2, color='b') # dispersed
            ax.fill_between(t_vals, scaled_root_v2 - scaled_std_root_v2, scaled_root_v2 + scaled_std_root_v2, color='b', alpha=0.3)
            ax.set_xlim(-0.02, 0.05)
            ax.set_ylim(-7, 70)
            # ax.set_yticks([-10, 30, 70])
            ax.hlines([-9, 30, 70], -0.02, 0.05, color='k', linestyle='--', linewidth=1)
            # add text to show max of root_v1 and v2
            ax.text(0.5, 0.6, f'r:{np.max(scaled_root_v1):.1f},b:{np.max(scaled_root_v2):.1f}', 
                    transform=ax.transAxes, va='center', ha='center', fontsize=6, fontweight='demibold')

            if j != 0:  # Not the leftmost column
                ax.set_yticks([])  # Remove y-axis ticks
                ax.spines["left"].set_visible(False)  # Remove left spine
            if i != 0:  # Not the bottom row
                ax.set_xticks([])  # Remove x-axis ticks
                ax.spines["bottom"].set_visible(False)  # Remove bottom spine

            if j == 0:
                ax.set_ylabel(f'{num_syn_per_clus[i]} synapses')
            if i == 0:
                ax.set_xlabel(f'{num_clus[j]} clusters')
            
            # Set label for two atrributes
            if i == 0 and j == num_cols-1:
                legend_patches = [
                Patch(facecolor='red', edgecolor='red', label=attr1, alpha=0.5),
                Patch(facecolor='blue', edgecolor='blue', label=attr2, alpha=0.5)]
                ax.legend(handles=legend_patches, loc='upper right', fontsize=6, frameon=True)

             # Remove upper and right spines (bounds)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{res_path}/{title}_root_v.jpg')

def full_analyze_case(simu_cond1='clus_invitro_variedW', simu_cond2='distr_invitro_variedW', anal_loc='basal', record_loc='Soma', record_attr='Peak', range=0, num_epochs=20):
    # Clustered case
    clus_mag_val_mat, clus_root_v_mat, clus_std_root_v_mat = analyze_case(root_folder_path, anal_loc+f'_range{range}_'+simu_cond1, num_syn_per_clus, num_clus, num_epochs, simu_cond1+' '+anal_loc+' (Range 0)', record_loc, record_attr)
    # Dispersed case
    distr_mag_val_mat, distr_root_v_mat, distr_std_root_v_mat = analyze_case(root_folder_path, anal_loc+f'_range{range}_'+simu_cond2, num_syn_per_clus, num_clus, num_epochs, simu_cond2+' '+anal_loc+' (Range 0)', record_loc, record_attr)
 
    return clus_mag_val_mat, distr_mag_val_mat, \
           clus_root_v_mat, distr_root_v_mat, \
           clus_std_root_v_mat, distr_std_root_v_mat

# Define parameters
num_clus = [1, 2, 3, 6, 9, 18] 
num_syn_per_clus = [1, 3, 6, 12, 24, 48, 72]  

anal_loc, rec_attr = 'apical', 'Area'
attr1, attr2 = 'clus', 'distr'

for rec_loc in ['Nexus', 'Dend']:
    for range_idx in [2]:
        globals()[f'attr1_dend_matrix_{range_idx}_area'], globals()[f'attr2_dend_matrix_{range_idx}_area'], \
        globals()[f'attr1_dend_root_v_{range_idx}_area'], globals()[f'attr2_dend_root_v_{range_idx}_area'], \
        globals()[f'attr1_dend_std_root_v_{range_idx}_area'], globals()[f'attr2_dend_std_root_v_{range_idx}_area'] = full_analyze_case(f'clus_invitro_variedW_tau90_addNaK_overallbranchtest', 
                                                                                                                                    f'distr_invitro_variedW_tau90_addNaK_overallbranchtest', 
                                                                                                                                    anal_loc, rec_loc, rec_attr, range=range_idx, num_epochs=10)                                                                                                                           
    # Calculate ratio and draw heatmap
    for range_idx in [2]:
        calculate_ratio_and_draw_heatmap(globals()[f'attr1_dend_matrix_{range_idx}_area'], globals()[f'attr2_dend_matrix_{range_idx}_area'], 
                                         globals()[f'attr1_dend_root_v_{range_idx}_area'], globals()[f'attr2_dend_root_v_{range_idx}_area'], 
                                         globals()[f'attr1_dend_std_root_v_{range_idx}_area'], globals()[f'attr2_dend_std_root_v_{range_idx}_area'], 
                                         f'Invitro {anal_loc} {rec_loc} Range {range_idx} {rec_attr} VariedW Tau90 AddNaK',
                                         attr1, attr2)
    

