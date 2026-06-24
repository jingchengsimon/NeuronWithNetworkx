import numpy as np

def generate_indices(rnd, num_clusters, num_conn_per_preunit, num_preunit):
    results = [] 
    indices = []
    connections_per_cluster = [0] * num_clusters

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
        
    # Without replacement, each preunit will only contact each cluster once (this one)
    # With replacement, each preunit may contact each cluster multiple times
    for i in range(num_clusters):
        index_list = [j for j, lst in enumerate(results) for element in lst if element == i]
        # index_list = [j for j, lst in enumerate(results) if i in lst]
        indices.append(index_list)

    return indices

def generate_vecstim(rnd, pre_unit_ids, num_stim, stim_time, stim_time_var=5):
    spt_unit_list = []
    
    for _ in pre_unit_ids:
        spt_unit = np.floor(rnd.normal(loc=stim_time, scale=stim_time_var, size=num_stim)) # Varied stimulus time
        spt_unit_list.append(spt_unit)

    spt_unit_array = np.array(list(zip(pre_unit_ids, spt_unit_list)), dtype=[('pre_unit_id', int), ('spt_unit', object)])

    return spt_unit_array
