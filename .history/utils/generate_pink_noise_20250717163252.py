import numpy as np
from scipy import signal as ss
from scipy.stats import zscore
import matplotlib.pyplot as plt

def make_noise(num_traces=10, num_samples=1000, scale=2):
    np.random.seed(42)
    
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

    for i in np.arange(0,num_traces):
        # Create White Noise
        wn = np.random.normal(loc=1, scale=scale, size=num_samples)  # scale=0.5
        # Create '1/f' Noise
        invfn[i,:] = zscore(ss.lfilter(B, A, wn))                           

    return invfn[:,2000:]

for scale in [0.5,1,2]:
    sample_list = make_noise(num_traces=10,num_samples=6000, scale=scale)

    sample = sample_list[2]
    print(np.mean(sample))
    # plt.figure()
    # plt.plot(sample)

    sample[sample<0] = 0
    sample = sample/np.mean(sample)
    print(np.mean(sample))
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 10)
    plt.plot(sample)

plt.show()
