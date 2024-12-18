import pandas as pd
from itertools import product

def generate_simu_params():

    NUM_SYN_BASAL_EXC = [10042] # [10042, 10042-4000, 10042-8000]
    NUM_SYN_APIC_EXC = [16070] # [16070, 16070-6400, 16070-12800]
    NUM_SYN_BASAL_INH = [1023] # 1023
    NUM_SYN_APIC_INH = [1637] # 1637
    NUM_SYN_SOMA_INH = [150]

    DURATION = [1000] # 1000 ms
    
    simu_condition = ['invitro']
    spat_condition = ['clus'] # clus or distr

    basal_channel_type = ['AMPANMDA']
    sec_type = ['basal', 'apical'] 
    dis_to_root = [1]
    cluster_radius =  [5]

    bg_exc_freq = [2] # basal
    bg_inh_freq = [4] # basal the ratio is 1:4 for both basal and apical
    input_ratio_basal_apic = [6]
    bg_exc_channel_type = ['AMPANMDA']
    initW = [0.1]
    inh_delay = [4] # 4 ms

    num_of_stim = [1]
    stim_time = [500] # 2500 ms

    num_clusters = [1, 2, 3, 6, 9, 18, 24] # 5 This number just approximates the distribution of clusters from the same source
    num_conn_per_preunit = [3] # 3
    num_preunit_per_clus = [1, 2, 4, 8, 16, 24, 32] # 80 * 5 / 10 = 40 syn per cluster
    
    pref_ori_dg = [0]
    num_trials = [1]

    # 生成所有可能的A和B的组合
    combinations = list(product(NUM_SYN_BASAL_EXC,
                                NUM_SYN_APIC_EXC,
                                NUM_SYN_BASAL_INH,
                                NUM_SYN_APIC_INH,
                                NUM_SYN_SOMA_INH,
                                DURATION,
                                simu_condition, spat_condition, 
                                basal_channel_type, sec_type,
                                dis_to_root, num_clusters, cluster_radius, 
                                bg_exc_freq, bg_inh_freq, input_ratio_basal_apic, 
                                bg_exc_channel_type, initW, inh_delay,
                                num_of_stim, stim_time,
                                num_conn_per_preunit, num_preunit_per_clus,
                                pref_ori_dg, num_trials))

    # 创建DataFrame
    df = pd.DataFrame(combinations, columns=['NUM_SYN_BASAL_EXC',
                                            'NUM_SYN_APIC_EXC',
                                            'NUM_SYN_BASAL_INH',
                                            'NUM_SYN_APIC_INH',
                                            'NUM_SYN_SOMA_INH',
                                            'DURATION',
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
                                            'delay of inhibitory inputs',
                                            'number of stimuli',
                                            'time point of stimulation',
                                            'number of connection per preunit',
                                            'number of preunits per cluster',
                                            'pref_ori_dg',
                                            'num_trials'])
    
    df['folder_tag'] = (df.index + 1).astype(str)
    params_list = [{**df.iloc[i].to_dict()} for i in range(len(df))]

    return params_list

