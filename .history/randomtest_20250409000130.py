from numpy.random import default_rng
import numpy

rnd = default_rng(42)

initW = 0.0004 # 0.0004 uS = 0.4 nS
num_func_group = 10

sigma = 1
mu = np.log(initW) - 0.5*sigma**2
syn_w_distr = rnd.lognormal(mean=mu, sigma=sigma, size=50000)
         
initW_distr = rnd.choice(syn_w_distr, 1)[0]
                

num_func_group = num_func_group # (26,000/5)/100 = 52
pink_noise_array = make_noise(num_traces=num_func_group, num_samples=DURATION)
           
            
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
