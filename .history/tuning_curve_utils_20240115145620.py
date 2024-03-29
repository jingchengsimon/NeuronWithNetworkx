import numpy as np
from scipy.optimize import curve_fit
def calculate_cirvar(num_spikes, ori_dg_list):
        # 将角度转换为弧度
        theta_radians = np.deg2rad(ori_dg_list)

        # 计算 exp(2i * theta)
        complex_exp = np.exp(2j * theta_radians)

        # 计算加权和
        weighted_sum = np.sum(num_spikes * complex_exp) / np.sum(num_spikes)

        # 取模
        cirvar = round(np.abs(weighted_sum), 4)

        return cirvar

def calculate_osi(num_spikes, ori_dg_list, pref_ori_dg):
    
    r_pref = num_spikes[ori_dg_list.index(pref_ori_dg)] + num_spikes[ori_dg_list.index((pref_ori_dg + 180) % 360)]
    r_ortho = num_spikes[ori_dg_list.index((pref_ori_dg + 90)) % 360] + num_spikes[ori_dg_list.index((pref_ori_dg + 270) % 360)]

    osi = round((r_pref - r_ortho) / (r_pref + r_ortho), 4)
    
    return osi

def gaussian_function(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

def fit_gaussian(x, y):
    popt, _ = curve_fit(self.gaussian_function, x, y, p0=[1, np.mean(x), np.std(x)])
    return popt
