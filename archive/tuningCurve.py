import numpy as np
import time
import pandas as pd
import networkx as nx
from tqdm import tqdm
import warnings
import re
from math import floor
import matplotlib.pyplot as plt
import seaborn as sns
import numba 
from numba import jit
from scipy.ndimage import gaussian_filter1d
import glob
import os
from scipy.optimize import curve_fit

def gaussian_function(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

def fit_gaussian(x, y):
    popt, _ = curve_fit(gaussian_function, x, y, p0=[1, np.mean(x), np.std(x)], maxfev=2000)#, method='lm')
    return popt

def visualize_tuning_curve():
    pref_ori_dg = 0.0
    ori_dg_list = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    num_spikes = [23, 19, 13, 20, 24, 23, 17, 18]

    pref_ori_dg = 0
    r_pref = num_spikes[ori_dg_list.index(pref_ori_dg)] + num_spikes[ori_dg_list.index((pref_ori_dg + 180) % 360)]
    r_ortho = num_spikes[ori_dg_list.index((pref_ori_dg + 90)) % 360] + num_spikes[ori_dg_list.index((pref_ori_dg + 270) % 360)]

    OSI = (r_pref - r_ortho) / (r_pref + r_ortho)
    print('OSI of the model neuron: ', round(OSI, 4))
    # ori_dg_list = [(x - (pref_ori_dg - 180)) % 360 for x in ori_dg_list]
    # sorted_indices = np.argsort(ori_dg_list)
    # ori_dg_list = [ori_dg_list[i] for i in sorted_indices]

    # num_spikes = [num_spikes[i] for i in sorted_indices]
    # min_num_spikes = min(num_spikes)
    # num_spikes = [x - min_num_spikes for x in num_spikes]
    
    x_values = np.array(ori_dg_list)
    y_values = np.array(num_spikes)

    # Fit Gaussian curve
    # params = fit_gaussian(x_values, y_values)

    # Plot original data points
    plt.figure(figsize=(5, 5))
    plt.plot(x_values, y_values, label='Data Points')
    plt.show()

    # Plot Gaussian curve using the fitted parameters
    curve_x = np.linspace(min(x_values), max(x_values), 100)
    curve_y = gaussian_function(curve_x, *params)
    plt.plot(curve_x, curve_y, label='Gaussian Fit', color='red')

    # Mark the data points on the graph
    for i, txt in enumerate(num_spikes):
        plt.annotate(txt, (x_values[i], y_values[i]), textcoords="offset points", xytext=(0, 5), ha='center')

    # Display the plot
    plt.xlabel('Orientation (degrees)')
    plt.ylabel('Number of Spikes')
    plt.legend()  
    plt.show()

visualize_tuning_curve()