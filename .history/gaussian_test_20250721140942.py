
import numpy as np
import matplotlib.pyplot as plt

def gaussian_window(M, std):
    n = np.arange(0, M) - (M - 1.0) / 2
    return np.exp(-0.5 * (n / std) ** 2)

window = gaussian_window(51, std=7)
plt.plot(window)
plt.show()