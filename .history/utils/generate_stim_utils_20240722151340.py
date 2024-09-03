import glob
import pandas as pd
import numpy as np
from neuron import h

from neuron.units import mV
import matplotlib.pyplot as plt
import os

def generate_indices(rnd, num_clusters, num_conn_per_preunit, num_preunit):
    
    pref_ori_dg = 0
    unit_ids = np.arange(num_preunit)
    
    results = []  # 用于存储生成的列表
    indices = []
    for _ in range(len(unit_ids)):
        # choose 3 clusters without replacement
        sampled = rnd.choice(num_clusters, num_conn_per_preunit, replace=False)  
        results.append(sampled)

    # 查找包含从0到k-1的列表的索引
    for i in range(num_clusters):
        index_list = [j for j, lst in enumerate(results) if i in lst]
        indices.append(index_list)

    return pref_ori_dg, unit_ids, indices

def generate_vecstim(unit_ids, num_stim, folder_path):
    spt_unit_list = []
    
    for unit_id in unit_ids:
        ## Artificial spike trains
        # try:
        #     spt_unit = spt_grouped_df.get_group((unit_id, stim_id))
        #     spt_unit = (spt_unit['spike_time'].values - spt_unit['spike_time'].values[0]) * 1000
        # except KeyError:
        #     # for units not fired, add list of 0
        #     spt_unit = np.array([])

        ## Single/Double Netstim
        netstim = h.NetStim()
        netstim.number = num_stim
        netstim.interval = 10 # ms (the default value is actually 10 ms)
        # start after the simulation become stable, and also add random to the start time of clustered 
        # netstim.start = 500 
        netstim.start = np.random.normal(500, 5)
        netstim.noise = 0
        spt_unit = netstim

        spt_unit_list.append(spt_unit)

    # ## Check independence of spt_unit_list
    # stim_t = h.Vector()
    # stim_id = h.Vector()
    # for ns in spt_unit_list:
    #     nc = h.NetCon(ns, None)
    #     nc.record(stim_t, stim_id)

    # h.finitialize(-65 * mV)
    # h.continuerun(1000)

    # # show raster
    # plt.figure(figsize=(6, 6))
    # for i in range(len(spt_unit_list)):
    #     plt.vlines([t for t, id_ in zip(stim_t, stim_id) if id_ == i],
    #             i - 0.4, i + 0.4)
    # plt.yticks(range(len(spt_unit_list)))
    # plt.xlim(0, 1000)
    # plt.savefig(os.path.join(folder_path,'preunit_raster.png'))
        
    return spt_unit_list

def get_stim_ids(ori_dg):
    spt_path = 'C:/Users/Windows/Desktop/MIMOlab/Codes/AllenAtlas/results/spt_df'
    pref_ori_dg = 0
    session_id = 732592105

    # for calculate the OSI
    spt_file = glob.glob(spt_path + f'/{pref_ori_dg}_{session_id}/spt_{ori_dg}.csv')
    file_path = spt_file[0] # usually only one file
    spt_df = pd.read_csv(file_path, index_col=None, header=0)
    
    # we need the presynaptic units always the same
    stim_ids = np.sort(spt_df['stimulus_presentation_id'].unique())
    print(f'stim_ids: {stim_ids}')
    
    return stim_ids
