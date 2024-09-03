import numpy as np
from scipy import signal as ss
from scipy.stats import zscore

def make_noise(num_traces=100,num_samples=4999, mean_fr = 1, std_fr = 1):

    num_samples = num_samples+2000

    #fv = np.linspace(0, 1, 40);                                # Normalised Frequencies

    #a = 1/(1+2*fv);                                      # Amplitudes Of '1/f'

    #b = ss.firls(43, fv, a);                                   # Filter Numerator Coefficients

    B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]

    A = [1, -2.494956002,   2.017265875,  -0.522189400]

    invfn = np.zeros((num_traces,num_samples))

    for i in np.arange(0,num_traces):

        wn = np.random.normal(loc=1,

               scale=0.5,size=num_samples)

        invfn[i,:] = zscore(ss.lfilter(B, A, wn));                             # Create '1/f' Noise

    return invfn[:,2000:]

sample = make_noise(num_traces=10,num_samples=2000, mean_fr = 1, std_fr = 1)
print(sample[0])
