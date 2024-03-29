import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm

def add_single_synapse(section_df, section_synapse_df, 
                        num_syn, region, sim_type, sections_basal, sections_apical, rnd, lock):
    sections = sections_basal if region == 'basal' else sections_apical
    # e_syn, tau1, tau2, syn_weight = self.syn_param_exc if sim_type == 'exc' else self.syn_param_inh
    type = 'A' if sim_type == 'exc' else 'B'
    
    if region == 'basal':
        section_length = np.array(section_df.loc[section_df['section_type'] == 'dend', 'length'])  
    else:
        section_length = np.array(section_df.loc[section_df['section_type'] == 'apic', 'length'])

    def generate_synapse():
        section = random.choices(sections, weights=section_length)[0][0].sec
        section_name = section.psection()['name']
        
        section_id_synapse = section_df.loc[section_df['section_name'] == section_name, 'section_id'].values[0]

        loc = rnd.uniform()
        segment_synapse = section(loc)

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

        with lock:
            section_synapse_df = section_synapse_df.append(data_to_append, ignore_index=True)

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        list(tqdm(executor.map(generate_synapse, range(num_syn)), total=num_syn))

    return section_synapse_df
