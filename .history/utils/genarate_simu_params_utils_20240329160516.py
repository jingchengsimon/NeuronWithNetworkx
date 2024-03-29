import pandas as pd
from itertools import product

def generate_simu_params():

    basal_channel_type = ['AMPANMDA','AMPA']
    sec_type = ['basal', 'tuft']
    dis_to_soma = [3,4]
    num_clusters = [10,20]
    cluster_radius =  [5]
    bg_syn_freq = [1]
    num_of_stim = [1,2]
    # num_syn_per_cluster = [20]
    num_conn_per_preunit = [4]
    # num_preunit = [10,25,50,100]
    pref_ori_dg = [0]
    num_trials = [5]

    # 生成所有可能的A和B的组合
    combinations = list(product(basal_channel_type, sec_type,
                                dis_to_soma, num_clusters, 
                                cluster_radius, bg_syn_freq, 
                                num_of_stim, 
                                num_conn_per_preunit, 
                                pref_ori_dg, num_trials))

    # 创建DataFrame
    df = pd.DataFrame(combinations, columns=['basal channel type',
                                            'section type',
                                            'distance from basal clusters to soma',
                                            'number of clusters',
                                            'cluster radius',
                                            'background synapse frequency',
                                            'number of stimuli',
                                            'number of connection per preunit',
                                            'pref_ori_dg',
                                            'num_trials'])
    df['folder_tag'] = (df.index + 1).astype(str)

    param_common = {
            'NUM_SYN_BASAL_EXC': 10042,
            'NUM_SYN_APIC_EXC': 16070,
            'NUM_SYN_BASAL_INH': 1023,
            'NUM_SYN_APIC_INH': 1637,
        }

    params_list = [{**param_common, **df.iloc[i].to_dict()} for i in range(len(df))]

    return params_list

