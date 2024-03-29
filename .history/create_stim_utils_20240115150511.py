import glob
import pandas as pd
import numpy as np

def generate_indices(num_clusters, num_conn_per_preunit=3):
    
    spt_path = 'C:/Users/Windows/Desktop/MIMOlab/Codes/AllenAtlas/results/spt_df'
    pref_ori_dg = 0
    session_id = 732592105
    ori_dg = 0.0

    # for calculate the OSI
    spt_file = glob.glob(spt_path + f'/{pref_ori_dg}_{session_id}/spt_{ori_dg}.csv')
    file_path = spt_file[0] # usually only one file
    spt_df = pd.read_csv(file_path, index_col=None, header=0)
    
    # we need the presynaptic units always the same
    unit_ids = np.sort(spt_df['unit_id'].unique())

    results = []  # 用于存储生成的列表
    indices = []
    for _ in range(len(unit_ids)):
        # choose 3 clusters without replacement
        sampled = self.rnd.choice(num_clusters, num_conn_per_preunit, replace=False)  
        results.append(sampled)

    # 查找包含从0到k-1的列表的索引
    for i in range(num_clusters):
        index_list = [j for j, lst in enumerate(results) if i in lst]
        indices.append(index_list)

    return pref_ori_dg, unit_ids, indices

def _get_stim_ids(self, ori_dg):
    spt_path = 'C:/Users/Windows/Desktop/MIMOlab/Codes/AllenAtlas/results/spt_df'
    pref_ori_dg = 0
    self.pref_ori_dg = pref_ori_dg
    session_id = 732592105

    # for calculate the OSI
    spt_file = glob.glob(spt_path + f'/{pref_ori_dg}_{session_id}/spt_{ori_dg}.csv')
    file_path = spt_file[0] # usually only one file
    spt_df = pd.read_csv(file_path, index_col=None, header=0)
    
    # we need the presynaptic units always the same
    stim_ids = np.sort(spt_df['stimulus_presentation_id'].unique())
    print(f'stim_ids: {stim_ids}')
    
    return stim_ids


def create_vecstim(ori_dg, stim_id, unit_ids):
        # define the path of all csv files for spike trains
        spt_path = 'C:/Users/Windows/Desktop/MIMOlab/Codes/AllenAtlas/results/spt_df'

        # use glob.glob to extract the csv files with the orientation wanted
        # read every file in this folder
        pref_ori_dg = 0
        session_id = 732592105

        spt_file = glob.glob(spt_path + f'/{pref_ori_dg}_{session_id}/spt_{ori_dg}.csv')

        # folder_path = f'/{pref_ori_dg}_{session_id}/spt_{ori_dg}.csv'
        # spt_file = os.path.join(spt_path, folder_path)

        # for file_path in spt_file:
        file_path = spt_file[0] # usually only one file
        spt_df = pd.read_csv(file_path, index_col=None, header=0)
        
        ## Following part will differ across different ori_dg
        
        # not all units spike at each stimulus presentation, 
        # we need choose the stim_id with the max number of units 
        spt_grouped_df = spt_df.groupby(['unit_id', 'stimulus_presentation_id'])
        # max_stim_id = spt_df.groupby('stimulus_presentation_id')['unit_id'].nunique().idxmax()
        
        spt_unit_list = []
        ori_dg_list = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        for unit_id in unit_ids:
            try:
                spt_unit = spt_grouped_df.get_group((unit_id, stim_id))
                spt_unit = (spt_unit['spike_time'].values - spt_unit['spike_time'].values[0]) * 1000
            except KeyError:
                # for units not fired, add list of 0
                spt_unit = np.array([])

            spt_unit_list.append(spt_unit)

        return spt_unit_list
 