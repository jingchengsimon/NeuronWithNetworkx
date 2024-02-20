from neuron import h
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import tqdm

def add_background_exc_inputs(section_synapse_df, syn_param_exc, spike_interval, lock):
    sec_syn_bg_exc_df = section_synapse_df[section_synapse_df['type'] == 'A']
    num_syn_background_exc = len(sec_syn_bg_exc_df)

    e_syn, tau1, tau2, syn_weight = syn_param_exc 

    

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        list(tqdm(executor.map(process_section, range(num_syn_background_exc)), total=num_syn_background_exc))

    return section_synapse_df
