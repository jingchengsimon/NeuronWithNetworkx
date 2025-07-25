import numpy as np
from scipy import signal as ss
from scipy.stats import zscore
import matplotlib.pyplot as plt

def make_noise(num_traces=10, num_samples=1000, scale=2):
    np.random.seed(1)
    
    num_samples = num_samples+2000
    # Normalised Frequencies
    # fv = np.linspace(0, 1, 40)
    # Amplitudes Of '1/f'                                
    # a = 1/(1+2*fv)                
    # Filter Numerator Coefficients              
    # b = ss.firls(43, fv, a)                                   

    B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    A = [1, -2.494956002,   2.017265875,  -0.522189400]

    invfn = np.zeros((num_traces,num_samples))

    for i in np.arange(0, num_traces):
        # Create White Noise
        wn = np.random.normal(loc=1, scale=scale, size=num_samples)  # scale=0.5
        # Create '1/f' Noise
        invfn[i,:] = zscore(ss.lfilter(B, A, wn))                          

    return invfn[:,2000:]

FREQ_EXC, DURATION = 1, 4000

for scale in [0.5,5]:
    sample_list = make_noise(num_traces=10, num_samples=DURATION, scale=scale)

    sample = sample_list[2]
    print(np.mean(sample))
    # plt.figure()
    # plt.plot(sample)

    sample[sample<0] = 0
    sample = sample/np.mean(sample)
    spk_rnd = np.random.default_rng(42)

    counts = spk_rnd.poisson(FREQ_EXC/1000 * sample)
    counts_ori = spk_rnd.poisson(FREQ_EXC/1000, DURATION)

    print(np.mean(sample))
    plt.subplots(figsize=(10, 4.5), nrows=4, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 0.5, 0.5, 0.5]})
    plt.subplot(4,1,1)
    plt.ylim(0, 5)
    plt.plot(sample)

    plt.subplot(4,1,2)
    plt.ylim(0, 1)
    plt.plot(counts_ori)

    plt.subplot(4,1,3)
    plt.ylim(0, 1)
    plt.plot(counts)

    plt.subplot(4,1,4)
    spike_train_bg = np.where(counts >= 1)[0] # ndarray
    mask = spk_rnd.choice([True, False], size=spike_train_bg.shape, p=[0.5, 0.5])
    spike_train_bg = spike_train_bg[mask]
    plt.vlines(spike_train_bg, 0, 1, color='C0')
    print(spike_train_bg)

    plt.tight_layout()

    # Remove top and right spines
    axs = plt.gcf().get_axes()
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

plt.show()
