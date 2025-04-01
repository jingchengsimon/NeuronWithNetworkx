import glob
import pandas as pd
import numpy as np
import time

def generate_indices(rnd, num_clusters, num_conn_per_preunit, num_preunit):
    
    pref_ori_dg = 0
    
    results = [] 
    indices = []
    connections_per_cluster = [0] * num_clusters

    # Choose num_conn clusters without replacement (replace=False)
    # for _ in range(num_preunit):    
    #     sampled = rnd.choice(num_clusters, num_conn_per_preunit, replace=False)  
    #     results.append(sampled)

    # Updated version
    # Round-robin approach to evenly distribute connections without replacement
    for _ in range(num_preunit):
        sampled = []
        available_clusters = list(range(num_clusters))
        for _ in range(num_conn_per_preunit):
            # Find the cluster with the minimum number of connections
            min_connections = min(connections_per_cluster)
            min_clusters = [i for i in available_clusters if connections_per_cluster[i] == min_connections]
            
            # Randomly choose one of the clusters with the minimum number of connections
            chosen_cluster = rnd.choice(min_clusters)
            
            # Add the chosen cluster to the sampled list and update the connection counter
            sampled.append(chosen_cluster)
            connections_per_cluster[chosen_cluster] += 1
            
            # Remove the chosen cluster from the available clusters
            available_clusters.remove(chosen_cluster)
        
        results.append(sampled)
        
    # Without replacement, each preunit will only contact each cluster once
    # With replacement, each preunit may contact each cluster multiple times
    for i in range(num_clusters):
        index_list = [j for j, lst in enumerate(results) for element in lst if element == i]
        # index_list = [j for j, lst in enumerate(results) if i in lst]
        indices.append(index_list)

    return pref_ori_dg, indices

def generate_vecstim(unit_ids, num_stim, stim_time, folder_path):
    spt_unit_list = []
    
    for _ in unit_ids:
        ## Artificial spike trains
        # try:
        #     spt_unit = spt_grouped_df.get_group((unit_id, stim_id))
        #     spt_unit = (spt_unit['spike_time'].values - spt_unit['spike_time'].values[0]) * 1000
        # except KeyError:
        #     # for units not fired, add list of 0
        #     spt_unit = np.array([])

        ## Single/Double Netstim
        
        # netstim = h.NetStim()
        # netstim.number = num_stim
        # netstim.interval = 3 # ms (the default value is actually 10 ms)
        # netstim.start = 500 #np.random.normal(500, 5)
        # netstim.noise = 0
        # spt_unit = netstim

        # Comment out only for test
        # np.random.seed(int(time.time())) # Reset the random number generator
        # spt_unit = np.floor(np.random.normal(stim_time, 5, num_stim)) # Varied stimulus time
        spt_unit = np.floor(np.array([stim_time] * num_stim)) # Fixed stimulus time
        spt_unit_list.append(spt_unit)
     
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
