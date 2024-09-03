import pandas as pd
from itertools import product

def generate_simu_params():

    NUM_SYN_BASAL_EXC = [10042] # [10042, 10042-4000, 10042-8000]
    NUM_SYN_APIC_EXC = [16070] # [16070, 16070-6400, 16070-12800]
    NUM_SYN_BASAL_INH = [1023] # 1023
    NUM_SYN_APIC_INH = [1637] # 1637
    NUM_SYN_SOMA_INH = [150]

    DURATION = [1000] # 5000 ms
    
    basal_channel_type = ['AMPANMDA']
    sec_type = ['basal', 'apical'] 
    dis_to_root = [2, 3, 4]
    cluster_radius =  [5]

    bg_exc_freq = [2] # basal
    bg_inh_freq = [4] # basal the ratio is 1:4 for both basal and apical
    input_ratio_basal_apic = [6]
    bg_exc_channel_type = ['AMPANMDA']
    initW = [0.1]
    inh_delay = [4] # 4 ms

    num_of_stim = [1]
    stim_time = [500] # 2500 ms

    num_clusters = [1] # 5
    num_conn_per_preunit = [1] # 4
    num_preunit = [1] # 80 * 5 / 5 = 80 syn per cluster
    
    pref_ori_dg = [0]
    num_trials = [1]

    # 生成所有可能的A和B的组合
    combinations = list(product(NUM_SYN_BASAL_EXC,
                                NUM_SYN_APIC_EXC,
                                NUM_SYN_BASAL_INH,
                                NUM_SYN_APIC_INH,
                                NUM_SYN_SOMA_INH,
                                DURATION,
                                basal_channel_type, sec_type,
                                dis_to_root, num_clusters, cluster_radius, 
                                bg_exc_freq, bg_inh_freq, input_ratio_basal_apic, 
                                bg_exc_channel_type, initW, inh_delay,
                                num_of_stim, stim_time,
                                num_conn_per_preunit, num_preunit,
                                pref_ori_dg, num_trials))

    # 创建DataFrame
    df = pd.DataFrame(combinations, columns=['NUM_SYN_BASAL_EXC',
                                            'NUM_SYN_APIC_EXC',
                                            'NUM_SYN_BASAL_INH',
                                            'NUM_SYN_APIC_INH',
                                            'NUM_SYN_SOMA_INH',
                                            'DURATION',
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

