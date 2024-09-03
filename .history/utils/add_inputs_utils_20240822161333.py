from neuron import h
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from functools import partial
from tqdm import tqdm
import time
import numpy as np
import json
from utils.synapses_models import AMPANMDA
from utils.generate_pink_noise import make_noise

def add_background_exc_inputs(section_synapse_df, syn_param_exc, DURATION, FREQ_EXC, 
                              input_ratio_basal_apic, bg_exc_channel_type, initW, lock):
    
    # use timestamp to seed the random number generator for each trial
    # timestamp = int(time.time())
    
    sec_syn_bg_exc_df = section_synapse_df[section_synapse_df['type'] == 'A']
    num_syn_background_exc = len(sec_syn_bg_exc_df)

    num_func_group = 52 # (26,000/5)/100 = 52
    pink_noise_array = make_noise(num_traces=num_func_group, num_samples=DURATION)

    basal_spike_trains = []
    apical_spike_trains = []
    
    num_preunit_exc_basal = int(len(sec_syn_bg_exc_df[sec_syn_bg_exc_df['region'] == 'basal'])/5)
    num_preunit_exc_apical = int(len(sec_syn_bg_exc_df[sec_syn_bg_exc_df['region'] == 'apical'])/5)

    def process_generate_basal_spike_train(i):
        pink_noise = pink_noise_array[np.random.randint(num_func_group)]
        pink_noise[pink_noise<0] = 0
        pink_noise = pink_noise/np.mean(pink_noise)
        counts = np.random.poisson(FREQ_EXC/1000 * pink_noise)
        spike_train = np.where(counts >= 1)[0] 
        basal_spike_trains.append(spike_train)

    def process_generate_apical_spike_train(i):
        pink_noise = pink_noise_array[np.random.randint(num_func_group)]
        pink_noise[pink_noise<0] = 0
        pink_noise = pink_noise/np.mean(pink_noise)
        counts = np.random.poisson(FREQ_EXC/(input_ratio_basal_apic*1000) * pink_noise)
        spike_train = np.where(counts >= 1)[0] 
        apical_spike_trains.append(spike_train)

    
    # spike_counts_basal_exc = np.random.poisson(FREQ_EXC/1000 * pink_noise, size=(int(num_preunit_exc_basal/5), DURATION))
    # spike_counts_apical_exc = np.random.poisson(FREQ_EXC/(input_ratio_basal_apic*1000) * pink_noise, size=(int(num_preunit_exc_apical/5), DURATION))
    
    def process_section(i):
        section = sec_syn_bg_exc_df.iloc[i]
        e_syn, tau1, tau2, syn_weight = syn_param_exc
        
        if section['synapse'] is None:
            
            if bg_exc_channel_type == 'Exp2Syn':
                synapse = h.Exp2Syn(sec_syn_bg_exc_df.iloc[i]['segment_synapse']) 
                synapse.e = e_syn
                synapse.tau1 = tau1
                synapse.tau2 = tau2
            
            elif bg_exc_channel_type == 'AMPANMDA':
                syn_params = json.load(open('./modelFile/AMPANMDA.json', 'r'))
                syn_params['initW'] = initW
                synapse = AMPANMDA(syn_params, section['loc'], section['section_synapse'], 'AMPANMDA')

        else:
            synapse = section['synapse']

        # Use np.random.poisson to generate the spike counts independently
        # Remember to divide by 1000 to get the rate per ms
        
        # Random choose a pink noise trace and rectify it to remove negative values and rescaled its mean to 1
        
        pink_noise = pink_noise_array[np.random.randint(num_func_group)]
        pink_noise[pink_noise<0] = 0
        pink_noise = pink_noise/np.mean(pink_noise)

        if section['region'] == 'basal':
            counts = np.random.poisson(FREQ_EXC/1000 * pink_noise)
        elif section['region'] == 'apical':
            counts = np.random.poisson(FREQ_EXC/(input_ratio_basal_apic*1000) * pink_noise)

        spike_train = np.where(counts >= 1)[0] 

        netstim = h.VecStim()
        netstim.play(h.Vector(spike_train))

        if section['netcon'] is not None:
            section['netcon'].weight[0] = 0

        netcon = h.NetCon(netstim, synapse)
        netcon.delay = 0
        netcon.weight[0] = syn_weight

        # with lock:
        if section['synapse'] is None:
            section_synapse_df.at[section.name, 'synapse'] = synapse
        section_synapse_df.at[section.name, 'netstim'] = netstim
        section_synapse_df.at[section.name, 'random'] = None
        section_synapse_df.at[section.name, 'netcon'] = netcon
        
        section_synapse_df.at[section.name, 'spike_train'] = list(spike_train) # generate an array to save spike_train for each synapse

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        list(tqdm(executor.map(process_section, range(num_syn_background_exc)), total=num_syn_background_exc))
        # executor.map(process_section, range(num_syn_background_exc))

    return section_synapse_df

def add_background_inh_inputs(section_synapse_df, syn_param_inh, DURATION, FREQ_INH, 
                              input_ratio_basal_apic, num_syn_inh_list, inh_delay, lock):  
    
    e_syn, tau1, tau2, syn_weight = syn_param_inh
    exc_types = ['A', 'C']

    spike_times = section_synapse_df[section_synapse_df['type'].isin(exc_types)]['spike_train']
    total_spikes = np.zeros(DURATION)  # 初始化总spike数数组

    for spike_times_vec in spike_times:
        try:
            if np.all(np.floor(spike_times_vec).astype(int) < DURATION):
                total_spikes[np.floor(spike_times_vec).astype(int)] += 1 # 向下取整
        except ValueError:
            continue

    # 计算每个时间点的平均firing rate (Hz, /s)

    ## there 2 approaches equal each other

    # firing_rates = total_spikes / (np.mean(total_spikes) * time_interval)
    # firing_rates_inh = firing_rates * FREQ_INH / np.mean(firing_rates)
    # lambda_array = firing_rates_inh * time_interval

    lambda_array = FREQ_INH * total_spikes / np.sum(total_spikes)
    
    num_syn_basal_inh, num_syn_apic_inh, num_syn_apic_inh = num_syn_inh_list

    spike_counts_basal_inh = np.random.poisson(lambda_array, size=(int(num_syn_basal_inh/1), DURATION))
    spike_counts_apic_inh = np.random.poisson(lambda_array/input_ratio_basal_apic, size=(int(num_syn_apic_inh/1), DURATION))
    spike_counts_inh = np.random.poisson(lambda_array, size=(int(num_syn_basal_inh/1), DURATION))

    sec_syn_bg_inh_df = section_synapse_df[section_synapse_df['type'] == 'B']

    def process_section(i):
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

        # counts = spike_counts_inh[i]
        counts = np.random.choice(spike_counts_inh)
        spike_train = np.where(counts >= 1)[0] 

        netstim = h.VecStim()
        netstim.play(h.Vector(spike_train))

        if section['netcon'] is not None:
            section['netcon'].weight[0] = 0

        netcon = h.NetCon(netstim, synapse)
        netcon.delay = inh_delay # ms
        netcon.weight[0] = syn_weight

        with lock:
            if section['synapse'] is None:
                section_synapse_df.at[section.name, 'synapse'] = synapse
            section_synapse_df.at[section.name, 'netstim'] = netstim
            section_synapse_df.at[section.name, 'netcon'] = netcon

            section_synapse_df.at[section.name, 'spike_train'] = spike_train

    for region in ['basal', 'apical', 'soma']:

        spike_counts_inh = spike_counts_basal_inh if region == 'basal' else spike_counts_apic_inh
        num_syn_background_inh = num_syn_basal_inh if region == 'basal' else num_syn_apic_inh
        sec_syn_inh_df = sec_syn_bg_inh_df[sec_syn_bg_inh_df['region'] == region]

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            # list(tqdm(executor.map(process_section, range(num_syn_background_inh)), total=num_syn_background_inh))
            executor.map(process_section, range(num_syn_background_inh))

    return section_synapse_df

def add_clustered_inputs(section_synapse_df, 
                         syn_param_exc, 
                         num_clusters, 
                         basal_channel_type, 
                         initW,
                         spt_unit_list, 
                         lock):  
    
    # sec_syn_clustered_df = section_synapse_df[section_synapse_df['type'] == 'C']
    # num_syn_clustered = len(sec_syn_clustered_df)

    syn_weight = syn_param_exc[-1]
    
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
                # spt_unit = h.NetStim()
                # spt_unit.number = 0
                spt_unit = np.array([])
            
            ## Single/Double Netstim
            # netstim = spt_unit
            netstim = h.VecStim()
            netstim.play(h.Vector(spt_unit))
            
            if section['synapse'] is None:
                syn_params = json.load(open('./modelFile/AMPANMDA.json', 'r'))
                syn_params['initW'] = initW
                synapse = AMPANMDA(syn_params, section['loc'], section['section_synapse'], basal_channel_type)
            else:
                synapse = section['synapse']

            ## turn off old netcons
            if section['netcon'] is not None:
                section['netcon'].weight[0] = 0
            
            netcon = h.NetCon(netstim, synapse) # netstim is always from the same unit with diff orientation
            netcon.delay = 0
            netcon.weight[0] = syn_weight
                
            if section['synapse'] is None:
                section_synapse_df.at[section.name, 'synapse'] = synapse
            section_synapse_df.at[section.name, 'netstim'] = netstim
            section_synapse_df.at[section.name, 'netcon'] = netcon

            section_synapse_df.at[section.name, 'spike_train'] = spt_unit

            # time.sleep(0.01)

    return section_synapse_df
    


    # def process_inner_task(i, sec_syn_clustered_df): 
    #     section = sec_syn_clustered_df.iloc[i]
    
    #     try:
    #         spt_unit = spt_unit_list[section['pre_unit_id']]
    #     except IndexError:
    #         spt_unit = h.NetStim()
    #         spt_unit.number = 0
    #         spt_unit.noise = 0
        
    #     # spt_unit_summed_list = spt_unit_summed_lists[section['cluster_id']]
        
    #     ## Artificial spike trains
    #     # spt_unit_vector = h.Vector(spt_unit)
    #     # netstim = h.VecStim()
    #     # netstim.play(spt_unit_vector)

    #     ## Single/Double Netstim
    #     netstim = spt_unit

    #     if section['synapse'] is None:
    #         syn_params = json.load(open('./modelFile/AMPANMDA.json', 'r'))
    #         synapse = AMPANMDA(syn_params, section['loc'], section['section_synapse'], basal_channel_type)        
    #     else:
    #         synapse = section['synapse']

    #     ## turn off old netcons
    #     if section['netcon'] is not None:
    #         section['netcon'].weight[0] = 0
        
    #     if i <= num_syn_to_get_input:
    #         netcon = h.NetCon(netstim, synapse) # netstim is always from the same unit with diff orientation
    #         netcon.delay = 0
    #         netcon.weight[0] = syn_weight
            
    #     with lock:
    #         if section['synapse'] is None:
    #             section_synapse_df.at[section.name, 'synapse'] = synapse
    #         section_synapse_df.at[section.name, 'netstim'] = netstim
    #         section_synapse_df.at[section.name, 'netcon'] = netcon

    # def process_outer_task(j):
        
    #     sec_syn_clustered_df = section_synapse_df[(section_synapse_df['type'] == 'C') & 
    #                                               (section_synapse_df['cluster_id'] == j)]
    #     num_syn_clustered = len(sec_syn_clustered_df)

    #     with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    #         process_inner_task_partial = partial(process_inner_task, sec_syn_clustered_df=sec_syn_clustered_df)
    #         executor.map(process_inner_task_partial, range(num_syn_clustered))
        
    # with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    #     executor.map(process_outer_task, range(num_clusters))
