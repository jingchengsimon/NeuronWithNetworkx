import pandas as pd
from itertools import product

def generate_simu_params():

    NUM_SYN_BASAL_EXC = [10042] #[10042, 10042-4000, 10042-8000]
    NUM_SYN_APIC_EXC = [6000] #[16070, 16070-6400, 16070-12800]
    NUM_SYN_BASAL_INH = [1023] # 1023
    NUM_SYN_APIC_INH = [1637] # 1637
    DURATION = [5000] # 1000 ms
    
    bg_exc_channel_type = ['AMPANMDA']
    bg_exc_freq = [1]
    bg_inh_freq = [4]
    initW = [0.1]
    inh_delay = [4] # 4 ms

    basal_channel_type = ['AMPANMDA']
    cluster_radius =  [5]
    sec_type = ['basal', 'apical']
    dis_to_root = [2, 3, 4]
    num_clusters = [20] # don't vary the num of clusters 
                        # because with the same num of presyn nodes, diff num of clusers means diff expansion of clusters
                        # e.g. 4*100 syns for 20 clusters, avg 20 syns per cluster; for 10 clusters, avg 40 syns per cluster
                        # the latter cluster is more expanded
    
    num_stim = [1, 2]
    num_conn_per_preunit = [4]
    num_preunit = [300]
    # num_syn_per_cluster = [20] # not pre-designed but decided by num_clusters, num_conn_per_preunit and num_preunit
    
    pref_ori_dg = [0]
    num_trials = [5]

    # 生成所有可能的A和B的组合
    combinations = list(product(NUM_SYN_BASAL_EXC,
                                NUM_SYN_APIC_EXC,
                                NUM_SYN_BASAL_INH,
                                NUM_SYN_APIC_INH,
                                DURATION,
                                basal_channel_type, sec_type,
                                dis_to_root, num_clusters, 
                                cluster_radius, 
                                bg_exc_freq, bg_inh_freq, 
                                bg_exc_channel_type, initW, inh_delay,
                                num_stim, 
                                num_conn_per_preunit, num_preunit,
                                pref_ori_dg, num_trials))

    # 创建DataFrame
    df = pd.DataFrame(combinations, columns=['NUM_SYN_BASAL_EXC',
                                            'NUM_SYN_APIC_EXC',
                                            'NUM_SYN_BASAL_INH',
                                            'NUM_SYN_APIC_INH',
                                            'DURATION',
                                            'basal channel type',
                                            'section type',
                                            'distance from clusters to root',
                                            'number of clusters',
                                            'cluster radius',
                                            'background excitatory frequency',
                                            'background inhibitory frequency',
                                            'background excitatory channel type',
                                            'initial weight of AMPANMDA synapses', 
                                            'delay of inhibitory inputs',
                                            'number of stimuli',
                                            'number of connection per preunit',
                                            'number of preunit',
                                            'pref_ori_dg',
                                            'num_trials'])
    df['folder_tag'] = (df.index + 1).astype(str)


    # param_common = {
    # 'NUM_SYN_BASAL_EXC': 10042,
    # 'NUM_SYN_APIC_EXC': 16070,
    # 'NUM_SYN_BASAL_INH': 1023,
    # 'NUM_SYN_APIC_INH': 1637,
    #     }

    params_list = [{**df.iloc[i].to_dict()} for i in range(len(df))]

    return params_list
