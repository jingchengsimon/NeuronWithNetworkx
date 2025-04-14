from numpy.random import default_rng


rnd = 
                    initW_distr = rnd.choice(syn_w_distr, 1)[0]
                

        else:
            synapse = section['synapse']

        if spat_condition == 'clus':
            # Use np.random.poisson to generate the spike counts independently
            # Remember to divide by 1000 to get the rate per ms
            
            # Random choose a pink noise trace and rectify it to remove negative values and rescaled its mean to 1
            
            pink_noise = pink_noise_array[rnd.integers(num_func_group)] #pink_noise_array[rnd.randint(num_func_group)]
            pink_noise[pink_noise<0] = 0
            pink_noise = pink_noise/np.mean(pink_noise)

            if section['region'] == 'basal':
                counts = rnd.poisson(FREQ_EXC/1000 * pink_noise)
                # counts = np.random.poisson(FREQ_EXC/1000, size=DURATION)
            elif section['region'] == 'apical':
                counts = rnd.poisson(FREQ_EXC/(input_ratio_basal_apic*1000) * pink_noise)
                # counts = np.random.poisson(FREQ_EXC/(input_ratio_basal_apic*1000), size=DURATION)

            spike_train = np.where(counts >= 1)[0] # ndarray

            # Filter spike_train to only include time points that do not exceed 1000
            # spike_train = spike_train[spike_train <= 1000]

            # update with failure probability p for presynaptic neuron spike trains (randomly dropout p*100% of each spike train)
            mask = rnd.choice([True, False], size=spike_train.shape, p=[0.5, 0.5])
            spike_train = spike_train[mask]
