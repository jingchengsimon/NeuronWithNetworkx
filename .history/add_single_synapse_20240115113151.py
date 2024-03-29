import numpy as np

def add_single_synapse(self, num_syn, region, sim_type):
        sections = self.sections_basal if region == 'basal' else self.sections_apical
        e_syn, tau1, tau2, syn_weight = self.syn_param_exc if sim_type == 'exc' else self.syn_param_inh
        type = 'A' if sim_type == 'exc' else 'B'
        
        if region == 'basal':
            section_length = np.array(self.section_df.loc[self.section_df['section_type'] == 'dend', 'length'])  
        else:
            section_length = np.array(self.section_df.loc[self.section_df['section_type'] == 'apic', 'length'])

        def generate_synapse(_):
            section = random.choices(sections, weights=section_length)[0][0].sec
            section_name = section.psection()['name']
            
            section_id_synapse = self.section_df.loc[self.section_df['section_name'] == section_name, 'section_id'].values[0]

            loc = self.rnd.uniform()
            segment_synapse = section(loc)
            
            # synapse = h.Exp2Syn(segment_synapse)
            # synapse.e = e_syn
            # synapse.tau1 = tau1
            # synapse.tau2 = tau2

            data_to_append = {'section_id_synapse': section_id_synapse,
                            'section_synapse': section,
                            'segment_synapse': segment_synapse,
                            'synapse': None, 
                            'netstim': None,
                            'random': None,
                            'netcon': None,
                            'loc': loc,
                            'type': type,
                            'cluster_id': -1,
                            'pre_unit_id': -1,
                            'region': region}

            with self.lock:
                self.section_synapse_df = self.section_synapse_df.append(data_to_append, ignore_index=True)

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            list(tqdm(executor.map(generate_synapse, range(num_syn)), total=num_syn))
