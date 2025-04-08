from neuron import h
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from functools import partial
from tqdm import tqdm
import time
import numpy as np
import json
import ast
from utils.synapses_models import AMPANMDA
from utils.generate_pink_noise import make_noise

def add_background_exc_inputs(section_synapse_df, syn_param_exc, DURATION, FREQ_EXC, 
                              input_ratio_basal_apic, bg_exc_channel_type, initW, num_func_group, rnd, epoch_idx, lock):
    
    # use timestamp to seed the random number generator for each trial
    # timestamp = int(time.time())
    
    sec_syn_bg_exc_df = section_synapse_df[section_synapse_df['type'] == 'A']
    num_syn_background_exc = len(sec_syn_bg_exc_df)

    num_func_group = num_func_group # (26,000/5)/100 = 52
    pink_noise_array = make_noise(num_traces=num_func_group, num_samples=DURATION)
    
    # Generate log-normal distribution
    # np.random.seed(42)  # For reproducibility
    sigma = 1
    mu = np.log(initW) - 0.5*sigma**2
    syn_w_distr = rnd.lognormal(mean=mu, sigma=sigma, size=50000)
    # np.random.seed(int(time.time())) # Reset the random number generator

    def process_section(i):

        # rnd = np.random.RandomState(int(time.time()))  # Create a new random state for each section
        rnd = np.random.RandomState(epoch_idx + i)  # Create a new random state for each section
        
        section = sec_syn_bg_exc_df.iloc[i]
        e_syn, tau1, tau2 = syn_param_exc
        
        if section['synapse'] is None:
            
            if bg_exc_channel_type == 'Exp2Syn':
                synapse = h.Exp2Syn(sec_syn_bg_exc_df.iloc[i]['segment_synapse']) 
                synapse.e = e_syn
                synapse.tau1 = tau1
                synapse.tau2 = tau2
            
            else:
                syn_params = json.load(open('./modelFile/AMPANMDA.json', 'r'))
                initW_distr = rnd.choice(syn_w_distr, 1)[0]
                syn_params['initW'] = initW_distr # variedW
                synapse = AMPANMDA(syn_params, section['loc'], section['section_synapse'], bg_exc_channel_type)

        else:
            synapse = section['synapse']

        # Use np.random.poisson to generate the spike counts independently
        # Remember to divide by 1000 to get the rate per ms
        
        # Random choose a pink noise trace and rectify it to remove negative values and rescaled its mean to 1
        
        pink_noise = pink_noise_array[rnd.randint(num_func_group)]
        pink_noise[pink_noise<0] = 0
        pink_noise = pink_noise/np.mean(pink_noise)

        if section['region'] == 'basal':
            counts = rnd.poisson(FREQ_EXC/1000 * pink_noise)
            # counts = np.random.poisson(FREQ_EXC/1000, size=DURATION)
        elif section['region'] == 'apical':
            counts = rnd.poisson(FREQ_EXC/(input_ratio_basal_apic*1000) * pink_noise)
            # counts = np.random.poisson(FREQ_EXC/(input_ratio_basal_apic*1000), size=DURATION)

        spike_train = np.where(counts >= 1)[0] # bdarray

        # Filter spike_train to only include time points that do not exceed 1000
        # spike_train = spike_train[spike_train <= 1000]

        # update with failure probability p for presynaptic neuron spike trains (randomly dropout p*100% of each spike train)
        mask = rnd.choice([True, False], size=spike_train.shape, p=[0.5, 0.5])
        spike_train = spike_train[mask]

        netstim = h.VecStim()
        netstim.play(h.Vector(spike_train))

        if section['netcon'] is not None:
            section['netcon'].weight[0] = 0

        netcon = h.NetCon(netstim, synapse)
        netcon.delay = 0
        netcon.weight[0] = 1 #syn_weight

        # with lock:
        if section['synapse'] is None:
            section_synapse_df.at[section.name, 'synapse'] = synapse
            section_synapse_df.at[section.name, 'syn_w'] = 1000 * initW_distr
        section_synapse_df.at[section.name, 'netstim'] = netstim
        section_synapse_df.at[section.name, 'spike_train'].append(list(spike_train)) 
        section_synapse_df.at[section.name, 'netcon'] = netcon
        
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        list(tqdm(executor.map(process_section, range(num_syn_background_exc)), total=num_syn_background_exc))
        # executor.map(process_section, range(num_syn_background_exc))

    return section_synapse_df

def add_background_inh_inputs(section_synapse_df, syn_param_inh, DURATION, FREQ_INH, 
                              input_ratio_basal_apic, num_syn_inh_list, inh_delay, rnd, epoch_idx, 
                              spat_condition, section_synapse_df_clus, lock):  
    
    # np.random.seed(42)  # For reproducibility

    e_syn, tau1, tau2, syn_weight = syn_param_inh
    exc_types = ['A', 'C']

    num_syn_basal_inh, num_syn_apic_inh, num_syn_soma_inh = num_syn_inh_list

    def spike_counts_generation(exc_types):
        spike_times = section_synapse_df[section_synapse_df['type'].isin(exc_types)]['spike_train'] 
        total_spikes = np.zeros(DURATION)  # 初始化总spike数数组

        for spike_times_vec in spike_times:
            spike_times_vec = spike_times_vec[0]  # 'spike train' is a list of lists (spike_trains)
            if spike_times_vec is not None and len(spike_times_vec) > 0:

                spike_times_vec = np.floor(spike_times_vec).astype(int)

                spike_times_vec = spike_times_vec[spike_times_vec < DURATION]
                total_spikes[spike_times_vec] += 1

        lambda_array = FREQ_INH * total_spikes / np.mean(total_spikes)

        spike_counts_basal_inh = rnd.poisson(lambda_array/1000, size=(int(num_syn_basal_inh/1), DURATION))
        spike_counts_apic_inh = rnd.poisson(lambda_array/(input_ratio_basal_apic*1000), size=(int(num_syn_apic_inh/1), DURATION))
        spike_counts_soma_inh = rnd.poisson(lambda_array/1000, size=(int(num_syn_soma_inh/1), DURATION))
    
        return spike_counts_basal_inh, spike_counts_apic_inh, spike_counts_soma_inh

    spike_counts_basal_inh, spike_counts_apic_inh, spike_counts_soma_inh = spike_counts_generation(exc_types)
    # spike_counts_basal_inh_bgexc, spike_counts_apic_inh_bgexc, spike_counts_soma_inh_bgexc = spike_counts_generation(['A'])
    # spike_counts_basal_inh_clusexc, spike_counts_apic_inh_clusexc, spike_counts_soma_inh_clusexc = spike_counts_generation(['C'])

    sec_syn_bg_inh_df = section_synapse_df[section_synapse_df['type'] == 'B']

    sec_syn_bg_inf_df_clus = section_synapse_df_clus[section_synapse_df_clus['type'] == 'B']

    def process_section(i):
        
        # rnd = np.random.RandomState(int(time.time()))  # Create a new random state for each section
        rnd = np.random.RandomState(epoch_idx + i)  # Create a new random state for each section

        section = sec_syn_inh_df.iloc[i]

        if section['synapse'] is None:
            synapse = h.Exp2Syn(sec_syn_bg_inh_df.iloc[i]['segment_synapse'])
            synapse.e = e_syn
            synapse.tau1 = tau1
            synapse.tau2 = tau2

        else:
            synapse = section['synapse']

        # generate the spike train in which the elements are time indices of each spike (with random noise)
        # e.g. the count is [0,1,0,0,1], then spike_train is [1, 4] with noise [0,1) ([1.1, 3.8])

        # A better to way to generate spike train is to repeat the time indices of each spike 
        # instead to just record the time point with spikes more than 1 
        # since we should repeat those time points with spikes mor than 2
        if spat_condition == 'clus':
            counts = spike_counts_inh[i]
            # counts = np.random.choice(spike_counts_inh) # this increase firing (don't know why)
            spike_train = np.where(counts >= 1)[0] 

            # counts_bgexc, counts_clusexc = spike_counts_inh_bgexc[i], spike_counts_inh_clusexc[i]
            # spike_train_inh_bgexc = np.where(counts_bgexc >= 1)[0]
            # spike_train_inh_clusexc = np.where(counts_clusexc >= 1)[0]

            # update with failure probability p for presynaptic neuron spike trains (randomly dropout p*100% of each spike train)
            mask = rnd.choice([True, False], size=spike_train.shape, p=[0.5, 0.5])
            spike_train = spike_train[mask]
        elif spat_condition == 'distr':
            section_clus = sec_syn_bg_inf_df_clus.iloc[i]
            spike_train = ast.literal_eval(section_clus['spike_train'])[0]

        netstim = h.VecStim()
        netstim.play(h.Vector(spike_train))

        if section['netcon'] is not None:
            section['netcon'].weight[0] = 0

        netcon = h.NetCon(netstim, synapse)
        netcon.delay = inh_delay # ms
        netcon.weight[0] = syn_weight

        # with lock:
        if section['synapse'] is None:
            section_synapse_df.at[section.name, 'synapse'] = synapse
            section_synapse_df.at[section.name, 'syn_w'] = 1000 * syn_weight
        section_synapse_df.at[section.name, 'netstim'] = netstim
        section_synapse_df.at[section.name, 'spike_train'].append(list(spike_train))
        # section_synapse_df.at[section.name, 'spike_train_inh_bgexc'].append(list(spike_train_inh_bgexc))
        # section_synapse_df.at[section.name, 'spike_train_inh_clusexc'].append(list(spike_train_inh_clusexc))
        section_synapse_df.at[section.name, 'netcon'] = netcon

    for region in ['basal', 'apical', 'soma']:

        if region == 'basal':
            spike_counts_inh = spike_counts_basal_inh
            # spike_counts_inh_bgexc, spike_counts_inh_clusexc = spike_counts_basal_inh_bgexc, spike_counts_basal_inh_clusexc
            num_syn_background_inh = num_syn_basal_inh
            sec_syn_inh_df = sec_syn_bg_inh_df[sec_syn_bg_inh_df['region'] == 'basal']

        elif region == 'apical':
            spike_counts_inh = spike_counts_apic_inh
            # spike_counts_inh_bgexc, spike_counts_inh_clusexc = spike_counts_apic_inh_bgexc, spike_counts_apic_inh_clusexc
            num_syn_background_inh = num_syn_apic_inh
            sec_syn_inh_df = sec_syn_bg_inh_df[sec_syn_bg_inh_df['region'] == 'apical']

        elif region == 'soma':
            spike_counts_inh = spike_counts_soma_inh
            # spike_counts_inh_bgexc, spike_counts_inh_clusexc = spike_counts_soma_inh_bgexc, spike_counts_soma_inh_clusexc
            num_syn_background_inh = num_syn_soma_inh
            sec_syn_inh_df = sec_syn_bg_inh_df[sec_syn_bg_inh_df['region'] == 'soma']

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            executor.map(process_section, range(num_syn_background_inh))

    return section_synapse_df

def add_clustered_inputs(section_synapse_df, 
                         simu_condition,
                         DURATION,
                         FREQ_EXC, 
                         input_ratio_basal_apic,
                         num_func_group,
                         num_clusters, 
                         basal_channel_type, 
                         initW,
                         spt_unit_list, 
                         rnd):  
    
    # sec_syn_clustered_df = section_synapse_df[section_synapse_df['type'] == 'C']
    # num_syn_clustered = len(sec_syn_clustered_df)

    # syn_weight = syn_param_exc[-1]
    
    # syn_params = json.load(open('./modelFile/AMPANMDA.json', 'r'))
    # syn_params['initW'] = initW
    
    # Generate log-normal distribution
    # np.random.seed(42)  # For reproducibility
    sigma = 1
    mu = np.log(initW) - 0.5*sigma**2
    syn_w_distr = rnd.lognormal(mean=mu, sigma=sigma, size=50000)
    # np.random.seed(int(time.time())) # Reset the random number generator

    num_func_group = num_func_group # (26,000/5)/100 = 52
    pink_noise_array = make_noise(num_traces=num_func_group, num_samples=DURATION)
    
    for j in range(num_clusters):

        sec_syn_clustered_df = section_synapse_df[(section_synapse_df['type'] == 'C') & 
                                                  (section_synapse_df['cluster_id'] == j)]
        num_syn_clustered = len(sec_syn_clustered_df)

        for i in range(num_syn_clustered):
            # need this change updated to the global dataframe
            section = sec_syn_clustered_df.iloc[i]

            try:
                spt_unit = spt_unit_list[section['pre_unit_id']]
            # spt_unit including pre_unit to the section is truncated 
            # or no preunit is connected to this section
            except IndexError:
                spt_unit = np.array([])

            if simu_condition == 'invivo':
                # bg part of spike train
                pink_noise = pink_noise_array[rnd.randint(num_func_group)]
                pink_noise[pink_noise<0] = 0
                pink_noise = pink_noise/np.mean(pink_noise)

                if section['region'] == 'basal':
                    counts = rnd.poisson(FREQ_EXC/1000 * pink_noise)
                    # counts = np.random.poisson(FREQ_EXC/1000, size=DURATION)
                elif section['region'] == 'apical':
                    counts = rnd.poisson(FREQ_EXC/(input_ratio_basal_apic*1000) * pink_noise)
                    # counts = np.random.poisson(FREQ_EXC/(input_ratio_basal_apic*1000), size=DURATION)

                spike_train = np.where(counts >= 1)[0] # bdarray

                mask = rnd.choice([True, False], size=spike_train.shape, p=[0.5, 0.5])
                spike_train_bg = spike_train[mask]

                spt_unit = spike_train_bg.tolist() + spt_unit

            ## Single/Double Netstim
            # netstim = spt_unit
            netstim = h.VecStim()
            netstim.play(h.Vector(spt_unit))

            # netstim.play(h.Vector(np.array([])))
            
            if section['synapse'] is None:
                syn_params = json.load(open('./modelFile/AMPANMDA.json', 'r'))
                # Comment out only for test
                initW_distr = rnd.choice(syn_w_distr, 1)[0]  # variedW
                # initW_distr = initW # fixedW
                syn_params['initW'] = initW_distr
                # print(f'initW_distr: {round(1000*initW_distr,3)}')
                synapse = AMPANMDA(syn_params, section['loc'], section['section_synapse'], basal_channel_type)
            else:
                synapse = section['synapse']

            ## turn off old netcons
            if section['netcon'] is not None:
                section['netcon'].weight[0] = 0
            
            netcon = h.NetCon(netstim, synapse) # netstim is always from the same unit with diff orientation
            netcon.delay = 0
            netcon.weight[0] = 1 #0.00016#syn_weight
                
            if section['synapse'] is None:
                section_synapse_df.at[section.name, 'synapse'] = synapse
                section_synapse_df.at[section.name, 'syn_w'] = 1000 * initW_distr # 1 uS = 1000 nS
            section_synapse_df.at[section.name, 'netstim'] = netstim
            section_synapse_df.at[section.name, 'spike_train'].append(spt_unit)
            section_synapse_df.at[section.name, 'netcon'] = netcon

    return section_synapse_df
    