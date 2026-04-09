import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import multiprocessing as mp
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

# 全局变量缓存
_data_cache = {}
_simu_info_cache = {}

def load_data_optimized(exp: str, use_cache: bool = True) -> Tuple:
    """
    优化的数据加载函数，支持缓存和并行加载
    """
    if use_cache and exp in _data_cache:
        return _data_cache[exp]
    
    root_folder_path = '/G/results/simulation/'
    
    # 预定义所有文件路径
    file_paths = {
        'v': os.path.join(root_folder_path, exp, 'dend_v_array.npy'),
        'i': os.path.join(root_folder_path, exp, 'dend_i_array.npy'),
        'nmda': os.path.join(root_folder_path, exp, 'dend_nmda_i_array.npy'),
        'ampa': os.path.join(root_folder_path, exp, 'dend_ampa_i_array.npy'),
        'nmda_g': os.path.join(root_folder_path, exp, 'dend_nmda_g_array.npy'),
        'ampa_g': os.path.join(root_folder_path, exp, 'dend_ampa_g_array.npy'),
        'soma': os.path.join(root_folder_path, exp, 'soma_v_array.npy'),
        'apic_v': os.path.join(root_folder_path, exp, 'apic_v_array.npy'),
        'apic_ica': os.path.join(root_folder_path, exp, 'apic_ica_array.npy'),
        'soma_i': os.path.join(root_folder_path, exp, 'soma_i_array.npy'),
        'trunk_v': os.path.join(root_folder_path, exp, 'trunk_v_array.npy'),
        'basal_v': os.path.join(root_folder_path, exp, 'basal_v_array.npy'),
        'tuft_v': os.path.join(root_folder_path, exp, 'tuft_v_array.npy'),
        'basal_bg_i_nmda': os.path.join(root_folder_path, exp, 'basal_bg_i_nmda_array.npy'),
        'basal_bg_i_ampa': os.path.join(root_folder_path, exp, 'basal_bg_i_ampa_array.npy'),
        'tuft_bg_i_nmda': os.path.join(root_folder_path, exp, 'tuft_bg_i_nmda_array.npy'),
        'tuft_bg_i_ampa': os.path.join(root_folder_path, exp, 'tuft_bg_i_ampa_array.npy')
    }
    
    def load_single_file(file_path: str) -> Optional[np.ndarray]:
        """并行加载单个文件"""
        try:
            return np.load(file_path)
        except FileNotFoundError:
            return None
    
    # 并行加载所有数据文件
    with ThreadPoolExecutor(max_workers=min(8, mp.cpu_count())) as executor:
        futures = {name: executor.submit(load_single_file, path) 
                  for name, path in file_paths.items()}
        
        data = {}
        for name, future in futures.items():
            result = future.result()
            if result is not None:
                data[name] = result
    
    dt = 1/40000
    
    # 加载JSON和CSV文件
    simu_info_path = os.path.join(root_folder_path, exp, 'simulation_params.json')
    sec_syn_df_path = os.path.join(root_folder_path, exp, 'section_synapse_df.csv')
    
    if exp not in _simu_info_cache:
        with open(simu_info_path) as f:
            _simu_info_cache[exp] = json.load(f)
    
    simu_info = _simu_info_cache[exp]
    sec_syn_df = pd.read_csv(sec_syn_df_path)
    
    # 构建返回元组
    result = (
        data.get('v'), data.get('i'), data.get('nmda'), data.get('ampa'),
        data.get('nmda_g'), data.get('ampa_g'), data.get('soma'), data.get('apic_v'),
        data.get('apic_ica'), data.get('soma_i'), data.get('trunk_v'), data.get('basal_v'),
        data.get('tuft_v'), data.get('basal_bg_i_nmda'), data.get('basal_bg_i_ampa'),
        data.get('tuft_bg_i_nmda'), data.get('tuft_bg_i_ampa'), dt, simu_info, sec_syn_df
    )
    
    if use_cache:
        _data_cache[exp] = result
    
    return result

def process_epoch_data(exp_path: str, epoch_idx: int, rec_loc: str, attr: str, 
                      t_start: int, t_end: int, dt: float) -> Dict:
    """
    处理单个epoch的数据，用于并行计算
    """
    try:
        v, i, nmda, ampa, nmda_g, ampa_g, soma, apic_v, apic_ica, soma_i, \
        trunk_v, basal_v, tuft_v, basal_bg_i_nmda, basal_bg_i_ampa, \
        tuft_bg_i_nmda, tuft_bg_i_ampa, dt, simu_info, sec_syn_df = load_data_optimized(exp_path, use_cache=False)
    except ValueError:
        v, i, nmda, ampa, nmda_g, ampa_g, soma, apic_v, apic_ica, dt, simu_info = load_data_optimized(exp_path, use_cache=False)
    
    # 数据预处理
    if v is not None and v.ndim == 5:
        v = np.mean(v, axis=2)
        soma = np.mean(soma, axis=1) if soma is not None else None
        apic_v = np.mean(apic_v, axis=1) if apic_v is not None else None
    
    # 计算基线
    v_base_trace = v[:,:,0,:].reshape(v.shape[0], v.shape[1], 1, v.shape[-1]) if v is not None else None
    soma_base_trace = soma[:,0,:].reshape(soma.shape[0], 1, soma.shape[-1]) if soma is not None else None
    apic_v_base_trace = apic_v[:,0,:].reshape(apic_v.shape[0], 1, apic_v.shape[-1]) if apic_v is not None else None
    
    # 计算EPSP
    EPSP_list = []
    x = np.arange(0, t_end-t_start)*dt
    
    if rec_loc == 'dend' and v is not None:
        if attr == 'peak':
            EPSP_list = np.mean(np.max(np.mean(v[:, t_start:t_end, :, :]-v_base_trace[:, t_start:t_end, :, :],axis=-1),axis=1),axis=0)
        elif attr == 'area':
            dend_over_baseline = np.mean(np.clip(np.mean(v[:, t_start:t_end, :, :]-v_base_trace[:, t_start:t_end, :, :],axis=-1), 1, None),axis=0)
            EPSP_list = np.trapz(dend_over_baseline, x, axis=0)
    
    elif rec_loc == 'soma' and soma is not None:
        if attr == 'peak':
            EPSP_list = np.max(np.mean(soma[t_start:t_end, :, :]-soma_base_trace[t_start:t_end, :, :],axis=-1),axis=0)
        elif attr == 'area':
            soma_over_baseline = np.clip(np.mean(soma[t_start:t_end, :, :]-soma_base_trace[t_start:t_end, :, :], axis=-1), 0, None)
            EPSP_list = np.trapz(soma_over_baseline, x, axis=0)
    
    elif rec_loc == 'nexus' and apic_v is not None:
        if attr == 'peak':
            EPSP_list = np.max(np.mean(apic_v[t_start:t_end, :, :]-apic_v_base_trace[t_start:t_end, :, :],axis=-1),axis=0)
        elif attr == 'area':
            apic_v_over_baseline = np.clip(np.mean(apic_v[t_start:t_end, :, :]-apic_v_base_trace[t_start:t_end, :, :], axis=-1), 0, None)
            EPSP_list = np.trapz(apic_v_over_baseline, x, axis=0)
    
    return {
        'epoch_idx': epoch_idx,
        'EPSP_list': EPSP_list,
        'v': v-v_base_trace if v is not None else None,
        'soma': soma-soma_base_trace if soma is not None else None,
        'apic_v': apic_v-apic_v_base_trace if apic_v is not None else None,
        'max_value': np.max(EPSP_list) if len(EPSP_list) > 0 else 0
    }

def nonlinearity_visualization_optimized(exp: str, ax_idx: int, exp_idx: int, fig, ax, 
                                       rec_loc: str, attr: str, plot_flag: bool, alpha: float = 1) -> Tuple:
    """
    优化的非线性可视化函数
    """
    # 使用缓存的数据加载
    v, i, nmda, ampa, nmda_g, ampa_g, soma, apic_v, apic_ica, soma_i, \
    trunk_v, basal_v, tuft_v, basal_bg_i_nmda, basal_bg_i_ampa, \
    tuft_bg_i_nmda, tuft_bg_i_ampa, dt, simu_info, sec_syn_df = load_data_optimized(exp)
    
    if v is not None and v.ndim == 5:
        v = np.mean(v, axis=2)
        i = np.mean(i, axis=2)
        nmda = np.mean(nmda, axis=2)
        ampa = np.mean(ampa, axis=2)
        nmda_g = np.mean(nmda_g, axis=2)
        ampa_g = np.mean(ampa_g, axis=2)
        soma = np.mean(soma, axis=1)
        apic_v = np.mean(apic_v, axis=1)

    t = simu_info['time point of stimulation']
    t_start, t_end = (t-20)*40, (t+100)*40
    
    # 基线计算
    v_base_trace = v[:,:,0,:].reshape(v.shape[0], v.shape[1], 1, v.shape[-1]) if v is not None else None
    soma_base_trace = soma[:,0,:].reshape(soma.shape[0], 1, soma.shape[-1]) if soma is not None else None
    apic_v_base_trace = apic_v[:,0,:].reshape(apic_v.shape[0], 1, apic_v.shape[-1]) if apic_v is not None else None
    
    EPSP_list = []

    if plot_flag:
        fig.subplots_adjust(wspace=0)
        
        syn_num_step = 1
        color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
        num_preunit, iter_step = 72, 2
        syn_num_list = list(range(0, num_preunit + 1, iter_step))

        if 'multiclus' in exp:
            syn_num_list = [0, 1, 3, 6, 12, 24, 48, 72]  
            
        ax.flat[ax_idx//syn_num_step].set_title(f'{exp_idx+1}')
        
        if rec_loc == 'dend' and v is not None:
            if attr == 'peak':
                EPSP_list = np.mean(np.max(np.mean(v[:, t_start:t_end, :, :]-v_base_trace[:, t_start:t_end, :, :],axis=-1),axis=1),axis=0)
            elif attr == 'area':
                x = np.arange(0, t_end-t_start)*dt
                dend_over_baseline = np.mean(np.clip(np.mean(v[:, t_start:t_end, :, :]-v_base_trace[:, t_start:t_end, :, :],axis=-1), 1, None),axis=0)
                EPSP_list = np.trapz(dend_over_baseline, x, axis=0)
            ax.flat[ax_idx//syn_num_step].plot(syn_num_list, EPSP_list, color=color_list[0], alpha=1)

        elif rec_loc == 'soma' and soma is not None:
            if attr == 'peak':
                EPSP_list = np.max(np.mean(soma[t_start:t_end, :, :]-soma_base_trace[t_start:t_end, :, :],axis=-1),axis=0)
            elif attr == 'area':
                x = np.arange(0, t_end-t_start)*dt
                soma_over_baseline = np.clip(np.mean(soma[t_start:t_end, :, :]-soma_base_trace[t_start:t_end, :, :], axis=-1), 0, None)
                EPSP_list = np.trapz(soma_over_baseline, x, axis=0)
            ax.flat[ax_idx//syn_num_step].plot(syn_num_list, EPSP_list, color='k', alpha=1)

        elif rec_loc == 'nexus' and apic_v is not None:
            if attr == 'peak':
                EPSP_list = np.max(np.mean(apic_v[t_start:t_end, :, :]-apic_v_base_trace[t_start:t_end, :, :],axis=-1),axis=0)
            elif attr == 'area':
                x = np.arange(0, t_end-t_start)*dt
                apic_v_over_baseline = np.clip(np.mean(apic_v[t_start:t_end, :, :]-apic_v_base_trace[t_start:t_end, :, :], axis=-1), 0, None)
                EPSP_list = np.trapz(apic_v_over_baseline, x, axis=0)
            ax.flat[ax_idx//syn_num_step].plot(syn_num_list, EPSP_list, color='b', alpha=1)

    return (v-v_base_trace if v is not None else None, 
            soma-soma_base_trace if soma is not None else None, 
            apic_v-apic_v_base_trace if apic_v is not None else None, 
            EPSP_list)

def full_nonlinearity_visualization_optimized(exp_list: List[str], idx_list: List[int], 
                                            rec_loc_list: List[str], attr_list: List[str], 
                                            num_epochs: int = 10, use_parallel: bool = True) -> Tuple:
    """
    优化的完整非线性可视化函数
    """
    t, dt = 500, 1/40000
    t_start, t_end = (t-20)*40, (t+100)*40
    
    # 创建图形
    num_ax_rows = np.ceil(num_epochs/10).astype(int)
    num_subplot_per_row = np.ceil(num_epochs/num_ax_rows).astype(int)
    
    fig, ax = plt.subplots(num_ax_rows, 1+num_subplot_per_row, 
                          figsize=(3*(1+num_subplot_per_row), 4*num_ax_rows), sharey=False)
    plt.suptitle(f'{exp_list[0]} {idx_list[0]} {rec_loc_list[0]}', fontsize=18)
    
    rec_loc, attr = rec_loc_list[0], attr_list[0]
    
    if use_parallel:
        # 并行处理所有epoch数据
        exp_path = f"{exp_list[0]}/{idx_list[0]}/"
        epoch_paths = [f"{exp_path}{epoch_idx + 1}/" for epoch_idx in range(num_epochs)]
        
        with ProcessPoolExecutor(max_workers=min(8, mp.cpu_count())) as executor:
            futures = [executor.submit(process_epoch_data, epoch_path, epoch_idx, rec_loc, attr, 
                                     t_start, t_end, dt) 
                      for epoch_idx, epoch_path in enumerate(epoch_paths)]
            
            epoch_results = [future.result() for future in futures]
    else:
        # 串行处理
        epoch_results = []
        for epoch_idx in range(num_epochs):
            exp_path = f"{exp_list[0]}/{idx_list[0]}/{epoch_idx + 1}/"
            result = process_epoch_data(exp_path, epoch_idx, rec_loc, attr, t_start, t_end, dt)
            epoch_results.append(result)
    
    # 根据最大EPSP值排序
    max_values = [result['max_value'] for result in epoch_results]
    sorted_indices = np.argsort(max_values)[::-1]
    print(f"排序后的epoch索引: {list(sorted_indices+1)}")
    
    # 重新绘制排序后的数据
    EPSP_list_list = []
    v_list, soma_list, apic_v_list = [], [], []
    
    for plot_idx, sorted_epoch_idx in enumerate(sorted_indices):
        result = epoch_results[sorted_epoch_idx]
        
        # 绘制单个epoch的数据
        exp_path = f"{exp_list[0]}/{idx_list[0]}/{sorted_epoch_idx + 1}/"
        v, soma, apic_v, EPSP_list = nonlinearity_visualization_optimized(
            exp_path, plot_idx, sorted_epoch_idx, fig, ax, rec_loc, attr, plot_flag=True)
        
        EPSP_list_list.append(result['EPSP_list'])
        
        if v is not None:
            v_list.append(np.mean(v, axis=(0, -1)))
        if soma is not None:
            soma_list.append(np.mean(soma, axis=-1))
        if apic_v is not None:
            apic_v_list.append(np.mean(apic_v, axis=-1))
    
    # 计算平均值和标准差
    EPSP_array = np.array(EPSP_list_list)
    avg_EPSP_list = np.mean(EPSP_array, axis=0)
    std_EPSP_list = np.std(EPSP_array, axis=0)
    
    # 绘制平均值
    num_preunit, iter_step, syn_num_step = 72, 2, 1
    syn_num_list = list(range(0, num_preunit + 1, iter_step))
    if 'multiclus' in exp_list[0]:
        syn_num_list = [0, 1, 3, 6, 12, 24, 48, 72]
    
    color = {'dend': 'C0', 'soma': 'k', 'nexus': 'b'}.get(rec_loc, 'k')
    
    ax.flat[-1//syn_num_step].plot(syn_num_list, avg_EPSP_list, color=color, alpha=1)
    ax.flat[-1//syn_num_step].fill_between(syn_num_list, 
                                          avg_EPSP_list - std_EPSP_list, 
                                          avg_EPSP_list + std_EPSP_list, 
                                          color=color, alpha=0.2)
    ax.flat[-1//syn_num_step].set_title('avg')
    
    # 设置图形属性
    max_ylim_peak, max_ylim_area = (80, 8) if rec_loc == 'dend' else (8, 0.4)
    
    for ax_idx in range(num_ax_rows*(1+num_subplot_per_row)):
        ax.flat[ax_idx//syn_num_step].set_xlabel('Number of Synapses')
        ax.flat[ax_idx//syn_num_step].set_xticks(list(range(0, num_preunit + 1, 12)))
        
        if attr == 'peak':
            ax.flat[ax_idx//syn_num_step].set_ylabel('EPSP (mV)')
            ax.flat[ax_idx//syn_num_step].set_ylim(-math.ceil(max_ylim_peak*1/16), 
                                                  math.ceil(max_ylim_peak*9/8))
            ax.flat[ax_idx//syn_num_step].set_yticks(list(range(0, int(max_ylim_peak*9/8), 
                                                                 int(max_ylim_peak*1/4))))
        elif attr == 'area':
            ax.flat[ax_idx//syn_num_step].set_ylabel('EPSP Area (mV ms)')
            ax.flat[ax_idx//syn_num_step].set_ylim(-max_ylim_area*1/16, max_ylim_area*9/8)
            ax.flat[ax_idx//syn_num_step].set_yticks(np.arange(0, max_ylim_area*9/8, max_ylim_area*1/4))
        
        # 移除顶部和右侧边框
        ax.flat[ax_idx//syn_num_step].spines['top'].set_visible(False)
        ax.flat[ax_idx//syn_num_step].spines['right'].set_visible(False)
    
    fig.tight_layout()
    
    return avg_EPSP_list, EPSP_list_list

# 清理缓存函数
def clear_cache():
    """清理数据缓存"""
    global _data_cache, _simu_info_cache
    _data_cache.clear()
    _simu_info_cache.clear()

# 使用示例
if __name__ == "__main__":
    # 示例调用
    exp_list = ['your_experiment_name']
    idx_list = [1]
    rec_loc_list = ['dend']
    attr_list = ['peak']
    
    # 使用优化版本
    avg_EPSP_list, EPSP_list_list = full_nonlinearity_visualization_optimized(
        exp_list, idx_list, rec_loc_list, attr_list, num_epochs=10, use_parallel=True
    )
    
    plt.show()
    
    # 清理缓存
    clear_cache() 