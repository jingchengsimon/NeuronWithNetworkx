import pandas as pd
from itertools import product

def generate_simu_params():

    NUM_SYN_BASAL_EXC = [10042] #[10042, 10042-4000, 10042-8000]
    NUM_SYN_APIC_EXC = [16070] #[16070, 16070-6400, 16070-12800]
    NUM_SYN_BASAL_INH = [1023]
    NUM_SYN_APIC_INH = [0]
    
    basal_channel_type = ['AMPANMDA']
    sec_type = ['basal']
    dis_to_root = [2]
    num_clusters = [1]
    cluster_radius =  [5]

    bg_exc_freq = [1]
    bg_inh_freq = [4]
    bg_exc_channel_type = ['AMPANMDA']
    initW = [0.05]

    num_of_stim = [1]
    # num_syn_per_cluster = [20] # not pre-designed but decided by num_clusters, num_conn_per_preunit and num_preunit
    num_conn_per_preunit = [4]
    num_preunit = [100]
    pref_ori_dg = [0]
    num_trials = [1]

    # 生成所有可能的A和B的组合
    combinations = list(product(NUM_SYN_BASAL_EXC,
                                NUM_SYN_APIC_EXC,
                                NUM_SYN_BASAL_INH,
                                NUM_SYN_APIC_INH,
                                basal_channel_type, sec_type,
                                dis_to_root, num_clusters, 
                                cluster_radius, 
                                bg_exc_freq, bg_inh_freq, 
                                bg_exc_channel_type, initW,
                                num_of_stim, 
                                num_conn_per_preunit, num_preunit,
                                pref_ori_dg, num_trials))

    # 创建DataFrame
    df = pd.DataFrame(combinations, columns=['NUM_SYN_BASAL_EXC',
                                            'NUM_SYN_APIC_EXC',
                                            'NUM_SYN_BASAL_INH',
                                            'NUM_SYN_APIC_INH',
                                            'basal channel type',
                                            'section type',
                                            'distance from clusters to root',
                                            'number of clusters',
                                            'cluster radius',
                                            'background excitatory frequency',
                                            'background inhibitory frequency',
                                            'background excitatory channel type',
                                            'initial weight of AMPANMDA synapses', 
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
