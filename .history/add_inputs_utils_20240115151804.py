from neuron import h
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm

def add_background_exc_inputs(section_synapse_df, syn_param_exc, spike_interval, lock):
    sec_syn_bg_exc_df = section_synapse_df[section_synapse_df['type'] == 'A']
    num_syn_background_exc = len(sec_syn_bg_exc_df)

     

    def process_section(i):
        section = sec_syn_bg_exc_df.iloc[i]
        e_syn, tau1, tau2, syn_weight = syn_param_exc

        if section['synapse'] is None:
            synapse = h.Exp2Syn(sec_syn_bg_exc_df.iloc[i]['segment_synapse'])
            synapse.e = e_syn
            synapse.tau1 = tau1
            synapse.tau2 = tau2
        else:
            synapse = section['synapse']

        netstim = h.NetStim()
        netstim.interval = spike_interval
        netstim.number = 10
        netstim.start = 0
        netstim.noise = 1

        random = h.Random()
        random.Random123(i)
        random.negexp(1)
        netstim.noiseFromRandom(random)

        if section['netcon'] is not None:
            section['netcon'].weight[0] = 0

        netcon = h.NetCon(netstim, synapse)
        netcon.delay = 0
        netcon.weight[0] = syn_weight

        with lock:
            if section['synapse'] is None:
                section_synapse_df.at[section.name, 'synapse'] = synapse
            section_synapse_df.at[section.name, 'netstim'] = netstim
            section_synapse_df.at[section.name, 'random'] = random
            section_synapse_df.at[section.name, 'netcon'] = netcon

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        list(tqdm(executor.map(process_section, range(num_syn_background_exc)), total=num_syn_background_exc))

    return section_synapse_df
