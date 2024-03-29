def _create_vecstim(self, ori_dg, stim_id, unit_ids):
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
 