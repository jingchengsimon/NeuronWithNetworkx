import pandas as pd
from itertools import product

def generate_simu_params(sec='basal', spat_cond='clus', dist=0):

    NUM_SYN_BASAL_EXC = [10042] # [10042, 10042-4000, 10042-8000]
    NUM_SYN_APIC_EXC = [16070] # [16070, 16070-6400, 16070-12800]
    NUM_SYN_BASAL_INH = [1023] # 1023
    NUM_SYN_APIC_INH = [1637] # 1637
    NUM_SYN_SOMA_INH = [150]

    SIMU_DURATION = [1000] # 1000 ms
    STIM_DURATION = [1000] # 1000 ms
    stim_time = [500] # 500 ms
    num_of_stim = [1]

    basal_channel_type = ['AMPA']
    bg_exc_channel_type = ['AMPA']
    cluster_radius =  [5] # 5 um

    simu_condition = ['invivo']
    spat_condition = [spat_cond] # clus or distr
    sec_type = [sec] # basal or apical
    dis_to_root = [dist]

    ## Invivo params
    bg_exc_freq = [1] # 1.3 basal ->0.5 1/4
    bg_inh_freq = [4] # 4 basal the ratio of exc/inh is 1:4 for both basal and apical
    input_ratio_basal_apic = [1]
    initW = [0.0004] # 0.0004 uS = 0.4 nS
    num_func_group = [10]
    inh_delay = [4] # 4 ms

    num_conn_per_preunit = [3] # 3
    num_clusters = [1]  # [1, 2, 3, 6, 9, 18] 
    num_syn_per_clus = [72]  # [1, 3, 6, 12, 24, 48, 72] [24, 32, 40, 48, 56, 64, 72]   
    
    pref_ori_dg = [0]
    num_trials = [1]

    # 生成所有可能的A和B的组合
    combinations = list(product(NUM_SYN_BASAL_EXC,
                                NUM_SYN_APIC_EXC,
                                NUM_SYN_BASAL_INH,
                                NUM_SYN_APIC_INH,
                                NUM_SYN_SOMA_INH,
                                SIMU_DURATION,
                                STIM_DURATION,
                                simu_condition, spat_condition, 
                                basal_channel_type, sec_type,
                                dis_to_root, num_clusters, cluster_radius, 
                                bg_exc_freq, bg_inh_freq, input_ratio_basal_apic, 
                                bg_exc_channel_type, initW, num_func_group, inh_delay,
                                num_of_stim, stim_time,
                                num_conn_per_preunit, num_syn_per_clus,
                                pref_ori_dg, num_trials))

    # 创建DataFrame
    df = pd.DataFrame(combinations, columns=['NUM_SYN_BASAL_EXC',
                                            'NUM_SYN_APIC_EXC',
                                            'NUM_SYN_BASAL_INH',
                                            'NUM_SYN_APIC_INH',
                                            'NUM_SYN_SOMA_INH',
                                            'SIMU DURATION',
                                            'STIM DURATION',
                                            'simulation condition',
                                            'synaptic spatial condition',
                                            'basal channel type',
                                            'section type',
                                            'distance from clusters to root',
                                            'number of clusters',
                                            'cluster radius',
                                            'background excitatory frequency',
                                            'background inhibitory frequency',
                                            'input ratio of basal to apical',
                                            'background excitatory channel type',
                                            'initial weight of AMPANMDA synapses',
                                            'number of functional groups', 
                                            'delay of inhibitory inputs',
                                            'number of stimuli',
                                            'time point of stimulation',
                                            'number of connection per preunit',
                                            'number of synapses per cluster',
                                            'pref_ori_dg',
                                            'num_trials'])
    
    df['folder_tag'] = (df.index + 1).astype(str)
    params_list = [{**df.iloc[i].to_dict()} for i in range(len(df))]

    return params_list

def generate_simu_params_REAL(sec='basal', spat_cond='clus', dist=0):

    NUM_SYN_BASAL_EXC = [10042] # [10042, 10042-4000, 10042-8000]
    NUM_SYN_APIC_EXC = [16070] # [16070, 16070-6400, 16070-12800]
    NUM_SYN_BASAL_INH = [1023] # 1023
    NUM_SYN_APIC_INH = [1637] # 1637
    NUM_SYN_SOMA_INH = [150]

    SIMU_DURATION = [6100] # 1000 ms
    STIM_DURATION = [6100] # 1000 ms
    stim_time = [500] # 500 ms
    num_of_stim = [1]

    basal_channel_type = ['AMPANMDA']
    bg_exc_channel_type = ['AMPANMDA']
    cluster_radius =  [5] # 5 um

    simu_condition = ['invivo']
    spat_condition = [spat_cond] # clus or distr
    sec_type = [sec] # basal or apical
    dis_to_root = [dist]

    ## Invivo params
    bg_exc_freq = [1.3] # 1.3 basal ->0.5 1/4
    bg_inh_freq = [4] # 4 basal the ratio of exc/inh is 1:4 for both basal and apical
    input_ratio_basal_apic = [1]
    initW = [0.0004] # 0.0004 uS = 0.4 nS
    num_func_group = [10]
    inh_delay = [4] # 4 ms

    num_conn_per_preunit = [3] # 3
    num_clusters = [1]  # [1, 2, 3, 6, 9, 18] 
    num_syn_per_clus = [1]  # [1, 3, 6, 12, 24, 48, 72] [24, 32, 40, 48, 56, 64, 72]   
    
    pref_ori_dg = [0]
    num_trials = [1]

    # 生成所有可能的A和B的组合
    combinations = list(product(NUM_SYN_BASAL_EXC,
                                NUM_SYN_APIC_EXC,
                                NUM_SYN_BASAL_INH,
                                NUM_SYN_APIC_INH,
                                NUM_SYN_SOMA_INH,
                                SIMU_DURATION,
                                STIM_DURATION,
                                simu_condition, spat_condition, 
                                basal_channel_type, sec_type,
                                dis_to_root, num_clusters, cluster_radius, 
                                bg_exc_freq, bg_inh_freq, input_ratio_basal_apic, 
                                bg_exc_channel_type, initW, num_func_group, inh_delay,
                                num_of_stim, stim_time,
                                num_conn_per_preunit, num_syn_per_clus,
                                pref_ori_dg, num_trials))

    # 创建DataFrame
    df = pd.DataFrame(combinations, columns=['NUM_SYN_BASAL_EXC',
                                            'NUM_SYN_APIC_EXC',
                                            'NUM_SYN_BASAL_INH',
                                            'NUM_SYN_APIC_INH',
                                            'NUM_SYN_SOMA_INH',
                                            'SIMU DURATION',
                                            'STIM DURATION',
                                            'simulation condition',
                                            'synaptic spatial condition',
                                            'basal channel type',
                                            'section type',
                                            'distance from clusters to root',
                                            'number of clusters',
                                            'cluster radius',
                                            'background excitatory frequency',
                                            'background inhibitory frequency',
                                            'input ratio of basal to apical',
                                            'background excitatory channel type',
                                            'initial weight of AMPANMDA synapses',
                                            'number of functional groups', 
                                            'delay of inhibitory inputs',
                                            'number of stimuli',
                                            'time point of stimulation',
                                            'number of connection per preunit',
                                            'number of synapses per cluster',
                                            'pref_ori_dg',
                                            'num_trials'])
    
    df['folder_tag'] = (df.index + 1).astype(str)
    params_list = [{**df.iloc[i].to_dict()} for i in range(len(df))]

    return params_list

