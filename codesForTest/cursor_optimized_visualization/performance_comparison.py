import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入原始函数（需要从你的notebook中复制）
def original_load_data(exp):
    """原始的数据加载函数"""
    root_folder_path = '/G/results/simulation/'
    
    try:
        v_path = [root_folder_path, exp, 'dend_v_array.npy']
        i_path = [root_folder_path, exp, 'dend_i_array.npy']
        nmda_path = [root_folder_path, exp, 'dend_nmda_i_array.npy']
        ampa_path = [root_folder_path, exp, 'dend_ampa_i_array.npy']
        
        nmda_g_path = [root_folder_path, exp, 'dend_nmda_g_array.npy']
        ampa_g_path = [root_folder_path, exp, 'dend_ampa_g_array.npy']
        
        soma_path = [root_folder_path, exp, 'soma_v_array.npy']
        apic_v_path = [root_folder_path, exp, 'apic_v_array.npy']
        apic_ica_path = [root_folder_path, exp, 'apic_ica_array.npy']

        soma_i_path = [root_folder_path, exp, 'soma_i_array.npy']
            
        trunk_v_path = [root_folder_path, exp, 'trunk_v_array.npy']
        basal_v_path = [root_folder_path, exp, 'basal_v_array.npy']
        tuft_v_path = [root_folder_path, exp, 'tuft_v_array.npy']

        basal_bg_i_nmda_path = [root_folder_path, exp, 'basal_bg_i_nmda_array.npy']
        basal_bg_i_ampa_path = [root_folder_path, exp, 'basal_bg_i_ampa_array.npy']
        tuft_bg_i_nmda_path = [root_folder_path, exp, 'tuft_bg_i_nmda_array.npy']
        tuft_bg_i_ampa_path = [root_folder_path, exp, 'tuft_bg_i_ampa_array.npy']

        v = np.load(os.path.join(*v_path))
        i = np.load(os.path.join(*i_path))
        nmda = np.load(os.path.join(*nmda_path))
        ampa = np.load(os.path.join(*ampa_path))
        
        nmda_g = np.load(os.path.join(*nmda_g_path))
        ampa_g = np.load(os.path.join(*ampa_g_path))
        
        soma = np.load(os.path.join(*soma_path))
        apic_v = np.load(os.path.join(*apic_v_path))
        apic_ica = np.load(os.path.join(*apic_ica_path))

        soma_i = np.load(os.path.join(*soma_i_path))

        trunk_v = np.load(os.path.join(*trunk_v_path))
        basal_v = np.load(os.path.join(*basal_v_path))
        tuft_v = np.load(os.path.join(*tuft_v_path))

        basal_bg_i_nmda = np.load(os.path.join(*basal_bg_i_nmda_path))
        basal_bg_i_ampa = np.load(os.path.join(*basal_bg_i_ampa_path))
        tuft_bg_i_nmda = np.load(os.path.join(*tuft_bg_i_nmda_path))
        tuft_bg_i_ampa = np.load(os.path.join(*tuft_bg_i_ampa_path)) 

    except FileNotFoundError:
        pass

    dt = 1/40000

    # read info from json and add to plot
    import json
    import pandas as pd
    with open(os.path.join(root_folder_path, exp, 'simulation_params.json')) as f:
        simu_info = json.load(f)
    
    with open(os.path.join(root_folder_path, exp, 'section_synapse_df.csv')) as f:
        sec_syn_df = pd.read_csv(f)

    try:
        return v, i, nmda, ampa, nmda_g, ampa_g, soma, apic_v, apic_ica, soma_i, \
               trunk_v, basal_v, tuft_v, basal_bg_i_nmda, basal_bg_i_ampa, \
               tuft_bg_i_nmda, tuft_bg_i_ampa, dt, simu_info, sec_syn_df
    
    except NameError:
        return simu_info, sec_syn_df

def original_nonlinearity_visualization(exp, ax_idx, exp_idx, fig, ax, rec_loc, attr, plot_flag, alpha=1):
    """原始的非线性可视化函数"""
    try:
        v, i, nmda, ampa, nmda_g, ampa_g, soma, apic_v, apic_ica, soma_i, \
        trunk_v, basal_v, tuft_v, basal_bg_i_nmda, basal_bg_i_ampa, \
        tuft_bg_i_nmda, tuft_bg_i_ampa, dt, simu_info, sec_syn_df = original_load_data(exp)
    
    except ValueError:
        v, i, nmda, ampa, nmda_g, ampa_g, soma, apic_v, apic_ica, dt, simu_info = original_load_data(exp)
    
    if v.ndim == 5:
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
    
    x = np.arange(0, t_end-t_start)*dt
    
    v_base_trace = v[:,:,0,:].reshape(v.shape[0], v.shape[1], 1, v.shape[-1])
    soma_base_trace = soma[:,0,:].reshape(soma.shape[0], 1, soma.shape[-1])
    apic_v_base_trace = apic_v[:,0,:].reshape(apic_v.shape[0], 1, apic_v.shape[-1])
    
    EPSP_list = []

    if plot_flag:
        fig.subplots_adjust(wspace=0)

        syn_num_step = 1
        color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
        num_clus_sampled = np.min([v.shape[0], 6])
        scale_factor_dend, scale_factor_root = 1, 1
        num_preunit, iter_step = 72, 2
        syn_num_list = list(range(0, num_preunit + 1, iter_step))

        if 'multiclus' in exp:
            syn_num_list = [0, 1, 3, 6, 12, 24, 48, 72]  
            
        ax.flat[ax_idx//syn_num_step].set_title(f'{exp_idx+1}')
        
        if rec_loc == 'dend':
            if attr == 'peak':
                EPSP_list = scale_factor_dend*np.mean(np.max(np.mean(v[:, t_start:t_end, :, :]-v_base_trace[:, t_start:t_end, :, :],axis=-1),axis=1),axis=0)
            elif attr == 'area':
                dend_over_baseline = scale_factor_dend*np.mean(np.clip(np.mean(v[:, t_start:t_end, :, :]-v_base_trace[:, t_start:t_end, :, :],axis=-1), 1, None),axis=0)
                EPSP_list = np.trapz(dend_over_baseline, x, axis=0)

            ax.flat[ax_idx//syn_num_step].plot(syn_num_list, EPSP_list, color=color_list[0], alpha=1)

        elif rec_loc == 'soma':
            if attr == 'peak':
                EPSP_list = scale_factor_root*np.max(np.mean(soma[t_start:t_end, :, :]-soma_base_trace[t_start:t_end, :, :],axis=-1),axis=0)
            elif attr == 'area':
                soma_over_baseline = scale_factor_root*np.clip(np.mean(soma[t_start:t_end, :, :]-soma_base_trace[t_start:t_end, :, :], axis=-1), 0, None)
                EPSP_list = np.trapz(soma_over_baseline, x, axis=0)
            
            ax.flat[ax_idx//syn_num_step].plot(syn_num_list, EPSP_list, color='k', alpha=1)

        elif rec_loc == 'nexus':
            if attr == 'peak':
                EPSP_list = scale_factor_root*np.max(np.mean(apic_v[t_start:t_end, :, :]-apic_v_base_trace[t_start:t_end, :, :],axis=-1),axis=0)
            elif attr == 'area':
                apic_v_over_baseline = scale_factor_root*np.clip(np.mean(apic_v[t_start:t_end, :, :]-apic_v_base_trace[t_start:t_end, :, :], axis=-1), 0, None)
                EPSP_list = np.trapz(apic_v_over_baseline, x, axis=0)

            ax.flat[ax_idx//syn_num_step].plot(syn_num_list, EPSP_list, color='b', alpha=1)

    return v-v_base_trace, soma-soma_base_trace, apic_v-apic_v_base_trace, EPSP_list

def original_full_nonlinearity_visualization(exp_list, idx_list, rec_loc_list, attr_list, num_epochs=10):
    """原始的完整非线性可视化函数"""
    t, dt = 500, 1/40000
    t_start, t_end = (t-20)*40, (t+100)*40
    t_vals = np.arange(t_start, t_end)*dt-t/1000
    
    num_figs = 1
    
    figs, axs = [], []

    num_ax_rows = np.ceil(num_epochs/10).astype(int)
    num_subplot_per_row = np.ceil(num_epochs/num_ax_rows).astype(int)
    for i in range(num_figs):
        fig, ax = plt.subplots(num_ax_rows, 1+num_subplot_per_row, figsize=(3*(1+num_subplot_per_row), 4*num_ax_rows), sharey=False)
        plt.suptitle(exp_list[0] + ' ' + str(idx_list[0]) + ' ' + rec_loc_list[0], fontsize=18)
        figs.append(fig)
        axs.append(ax)
    
    rec_loc, attr, ax = rec_loc_list[0], attr_list[0], axs[0]
    rec_loc_trace_list = []
    for epoch_idx in range(num_epochs):
        v, soma, apic_v, EPSP_list = original_nonlinearity_visualization(exp_list[0] + '/' + str(idx_list[0]) + '/' + str(epoch_idx + 1) + '/', 
                                epoch_idx, epoch_idx, figs[0], axs[0], rec_loc_list[0], attr_list[0], plot_flag=False)
        
        rec_loc_trace = {'dend': v, 'soma': soma, 'nexus': apic_v}.get(rec_loc)
        if rec_loc_trace is not None:
            rec_loc_trace_list.append(rec_loc_trace)
            
    if rec_loc == 'dend':
        max_values = [np.max(rec_loc_trace[:, t_start:t_end, -1, :]) for rec_loc_trace in rec_loc_trace_list]
    else:
        max_values = [np.max(rec_loc_trace[t_start:t_end, -1, :]) for rec_loc_trace in rec_loc_trace_list]

    sorted_indices = np.argsort(max_values)[::-1]
    print(list(sorted_indices+1))

    v_list, soma_list, apic_v_list, EPSP_list_list = [], [], [], []
    for epoch_idx in range(num_epochs):
        sorted_epoch_idx = sorted_indices[epoch_idx] 
        v, soma, apic_v, EPSP_list = original_nonlinearity_visualization(exp_list[0] + '/' + str(idx_list[0]) + '/' + str(sorted_epoch_idx + 1) + '/', 
                                epoch_idx, sorted_epoch_idx, figs[0], axs[0], rec_loc_list[0], attr_list[0], plot_flag=True)
        
        EPSP_list_list.append(EPSP_list)
        v_list.append(np.mean(v, axis=(0, -1)))
        soma_list.append(np.mean(soma, axis=-1))
        apic_v_list.append(np.mean(apic_v, axis=-1))
        
    num_preunit, iter_step, syn_num_step = 72, 2, 1
    syn_num_list = list(range(0, num_preunit + 1, iter_step))
    if 'multiclus' in exp_list[0]:
        syn_num_list = [0, 1, 3, 6, 12, 24, 48, 72] 
    
    avg_EPSP_list, std_EPSP_list = np.mean(np.array(EPSP_list_list), axis=0), np.std(np.array(EPSP_list_list), axis=0)
    aff_idx = -1
    print(f'Max avg EPSP: {round(avg_EPSP_list[aff_idx], 2)} mV')
    if rec_loc == 'dend':
        max_avg_EPSP = np.max(np.mean([v[t_start:t_end, aff_idx] for v in v_list], axis=0))
    elif rec_loc == 'soma':
        max_avg_EPSP = np.max(np.mean([soma[t_start:t_end, aff_idx] for soma in soma_list], axis=0))
    elif rec_loc == 'nexus':
        max_avg_EPSP = np.max(np.mean([apic_v[t_start:t_end, aff_idx] for apic_v in apic_v_list], axis=0))
    print(f'Max avg EPSP 2: {round(max_avg_EPSP, 2)} mV')
   
    color = {'dend': 'C0', 'soma': 'k', 'nexus': 'b'}.get(rec_loc, 'k')
    
    ax.flat[-1//syn_num_step].plot(syn_num_list, avg_EPSP_list, color=color, alpha=1)
    ax.flat[-1//syn_num_step].fill_between(syn_num_list, avg_EPSP_list - std_EPSP_list, avg_EPSP_list + std_EPSP_list, color=color, alpha=0.2)
    ax.flat[-1//syn_num_step].set_title(f'avg')

    if rec_loc == 'dend':
        max_ylim_peak, max_ylim_area = 80, 8
    elif rec_loc in ['soma', 'nexus']:
        max_ylim_peak, max_ylim_area = 8, 0.4

    import math
    for ax_idx in range(num_ax_rows*(1+num_subplot_per_row)):
        ax.flat[ax_idx//syn_num_step].set_xlabel('Number of Synapses')
        ax.flat[ax_idx//syn_num_step].set_xticks(list(range(0, num_preunit + 1, 12)))
        if attr == 'peak':
            ax.flat[ax_idx//syn_num_step].set_ylabel('EPSP (mV)')
        elif attr == 'area':
            ax.flat[ax_idx//syn_num_step].set_ylabel('EPSP Area (mV ms)')

        ax.flat[ax_idx//syn_num_step].spines['top'].set_visible(False)
        ax.flat[ax_idx//syn_num_step].spines['right'].set_visible(False)

        if attr == 'peak':
            ax.flat[ax_idx//syn_num_step].set_ylim(-math.ceil(max_ylim_peak*1/16), math.ceil(max_ylim_peak*9/8))
            ax.flat[ax_idx//syn_num_step].set_yticks(list(range(0, int(max_ylim_peak*9/8), int(max_ylim_peak*1/4))))
        elif attr == 'area':
            ax.flat[ax_idx//syn_num_step].set_ylim(-max_ylim_area*1/16, max_ylim_area*9/8)
            ax.flat[ax_idx//syn_num_step].set_yticks(np.arange(0, max_ylim_area*9/8, max_ylim_area*1/4))

    for fig in figs:
        fig.tight_layout()

    return avg_EPSP_list, EPSP_list_list

def performance_test():
    """性能测试函数"""
    # 测试参数
    exp_list = ['test_experiment']  # 替换为你的实际实验名称
    idx_list = [1]
    rec_loc_list = ['dend']
    attr_list = ['peak']
    num_epochs = 5  # 减少epoch数量以加快测试
    
    print("开始性能测试...")
    print(f"测试参数: {num_epochs} epochs, {rec_loc_list[0]}, {attr_list[0]}")
    
    # 测试原始版本
    print("\n=== 测试原始版本 ===")
    start_time = time.time()
    try:
        avg_EPSP_orig, EPSP_list_orig = original_full_nonlinearity_visualization(
            exp_list, idx_list, rec_loc_list, attr_list, num_epochs
        )
        original_time = time.time() - start_time
        print(f"原始版本执行时间: {original_time:.2f} 秒")
    except Exception as e:
        print(f"原始版本执行失败: {e}")
        original_time = float('inf')
    
    # 测试优化版本（串行）
    print("\n=== 测试优化版本（串行） ===")
    from optimized_visualization import full_nonlinearity_visualization_optimized, clear_cache
    clear_cache()  # 清理缓存
    
    start_time = time.time()
    try:
        avg_EPSP_opt_serial, EPSP_list_opt_serial = full_nonlinearity_visualization_optimized(
            exp_list, idx_list, rec_loc_list, attr_list, num_epochs, use_parallel=False
        )
        optimized_serial_time = time.time() - start_time
        print(f"优化版本（串行）执行时间: {optimized_serial_time:.2f} 秒")
    except Exception as e:
        print(f"优化版本（串行）执行失败: {e}")
        optimized_serial_time = float('inf')
    
    # 测试优化版本（并行）
    print("\n=== 测试优化版本（并行） ===")
    clear_cache()  # 清理缓存
    
    start_time = time.time()
    try:
        avg_EPSP_opt_parallel, EPSP_list_opt_parallel = full_nonlinearity_visualization_optimized(
            exp_list, idx_list, rec_loc_list, attr_list, num_epochs, use_parallel=True
        )
        optimized_parallel_time = time.time() - start_time
        print(f"优化版本（并行）执行时间: {optimized_parallel_time:.2f} 秒")
    except Exception as e:
        print(f"优化版本（并行）执行失败: {e}")
        optimized_parallel_time = float('inf')
    
    # 计算加速比
    print("\n=== 性能对比结果 ===")
    if original_time != float('inf') and optimized_serial_time != float('inf'):
        speedup_serial = original_time / optimized_serial_time
        print(f"串行优化加速比: {speedup_serial:.2f}x")
    
    if original_time != float('inf') and optimized_parallel_time != float('inf'):
        speedup_parallel = original_time / optimized_parallel_time
        print(f"并行优化加速比: {speedup_parallel:.2f}x")
    
    if optimized_serial_time != float('inf') and optimized_parallel_time != float('inf'):
        parallel_speedup = optimized_serial_time / optimized_parallel_time
        print(f"并行vs串行加速比: {parallel_speedup:.2f}x")
    
    # 清理缓存
    clear_cache()
    
    return {
        'original_time': original_time,
        'optimized_serial_time': optimized_serial_time,
        'optimized_parallel_time': optimized_parallel_time
    }

def benchmark_different_epochs():
    """测试不同epoch数量的性能"""
    exp_list = ['test_experiment']  # 替换为你的实际实验名称
    idx_list = [1]
    rec_loc_list = ['dend']
    attr_list = ['peak']
    
    epoch_counts = [1, 3, 5, 10]
    results = {}
    
    print("测试不同epoch数量的性能...")
    
    for num_epochs in epoch_counts:
        print(f"\n测试 {num_epochs} epochs...")
        
        # 测试原始版本
        start_time = time.time()
        try:
            original_full_nonlinearity_visualization(exp_list, idx_list, rec_loc_list, attr_list, num_epochs)
            original_time = time.time() - start_time
        except Exception as e:
            print(f"原始版本失败: {e}")
            original_time = float('inf')
        
        # 测试优化版本
        from optimized_visualization import full_nonlinearity_visualization_optimized, clear_cache
        clear_cache()
        
        start_time = time.time()
        try:
            full_nonlinearity_visualization_optimized(exp_list, idx_list, rec_loc_list, attr_list, num_epochs, use_parallel=True)
            optimized_time = time.time() - start_time
        except Exception as e:
            print(f"优化版本失败: {e}")
            optimized_time = float('inf')
        
        results[num_epochs] = {
            'original': original_time,
            'optimized': optimized_time,
            'speedup': original_time / optimized_time if optimized_time != float('inf') else 0
        }
        
        print(f"  {num_epochs} epochs - 原始: {original_time:.2f}s, 优化: {optimized_time:.2f}s, 加速比: {results[num_epochs]['speedup']:.2f}x")
    
    return results

if __name__ == "__main__":
    # 运行性能测试
    print("=== 可视化性能优化测试 ===\n")
    
    # 基本性能测试
    performance_results = performance_test()
    
    # 不同epoch数量的测试
    print("\n" + "="*50)
    epoch_results = benchmark_different_epochs()
    
    # 总结
    print("\n" + "="*50)
    print("优化总结:")
    print("1. 数据加载优化:")
    print("   - 并行文件加载")
    print("   - 数据缓存机制")
    print("   - 错误处理优化")
    
    print("\n2. 计算优化:")
    print("   - 并行epoch处理")
    print("   - 内存使用优化")
    print("   - 向量化计算")
    
    print("\n3. 预期加速效果:")
    print("   - 数据加载: 2-4x 加速")
    print("   - 并行处理: 2-8x 加速（取决于CPU核心数）")
    print("   - 总体加速: 3-10x 加速") 