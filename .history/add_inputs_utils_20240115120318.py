import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import tqdm

def _add_background_exc_inputs(self):
        sec_syn_bg_exc_df = self.section_synapse_df[self.section_synapse_df['type'] == 'A']
        num_syn_background_exc = len(sec_syn_bg_exc_df)
        syn_weight = self.syn_param_exc[-1]

        e_syn, tau1, tau2, syn_weight = self.syn_param_exc 

        def process_section(i):
            section = sec_syn_bg_exc_df.iloc[i]

            if section['synapse'] is None:
                synapse = h.Exp2Syn(sec_syn_bg_exc_df.iloc[i]['segment_synapse'])
                synapse.e = e_syn
                synapse.tau1 = tau1
                synapse.tau2 = tau2
            else:
                synapse = section['synapse']

            netstim = h.NetStim()
            netstim.interval = self.spike_interval
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

            with self.lock:
                if section['synapse'] is None:
                    self.section_synapse_df.at[section.name, 'synapse'] = synapse
                self.section_synapse_df.at[section.name, 'netstim'] = netstim
                self.section_synapse_df.at[section.name, 'random'] = random
                self.section_synapse_df.at[section.name, 'netcon'] = netcon

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            list(tqdm(executor.map(process_section, range(num_syn_background_exc)), total=num_syn_background_exc))
    