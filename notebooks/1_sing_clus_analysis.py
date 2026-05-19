import os
import json
import pandas as pd
import numpy as np
import warnings
import pickle
from datetime import datetime
import inspect
import re
import psutil
import gc
# ignore runtime warning
warnings.filterwarnings("ignore", category=RuntimeWarning)

# root_folder_path = '/G/results/visualization_simulation_singclus/' #'/mnt/mimo_1/simu_results_sjc/simulation_singclus_Aug25'  #'/G/results/simulation/'
root_folder_path = '/mnt/mimo_1/simu_results_sjc/simulation_singclus_Aug25'  #'/G/results/simulation/'

# 添加数据缓存以加速重复加载
_data_cache = {}

def check_memory_usage():
    """检查内存使用情况"""
    memory = psutil.virtual_memory()
    print(f"💾 Memory usage: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
    return memory.percent

def check_cpu_usage():
    """检查CPU使用情况"""
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"⚙️  CPU usage: {cpu_percent:.1f}%")
    return cpu_percent

def clear_memory(clear_globals=False, force_gc=True):
    """
    清理内存和资源
    
    Parameters:
    -----------
    clear_globals : bool
        是否清理所有EPSP相关的全局变量（谨慎使用，会删除已保存的结果）
    force_gc : bool
        是否强制执行多次垃圾回收
    """
    global _data_cache
    
    # 1. 清理数据缓存
    _data_cache.clear()
    
    # 2. 如果指定，清理EPSP相关的全局变量
    if clear_globals:
        to_delete = []
        for var_name in list(globals().keys()):
            if 'EPSP' in var_name or 'nmda_flag' in var_name:
                if not var_name.startswith('__'):
                    try:
                        # 先清除numpy数组引用
                        var = globals()[var_name]
                        if hasattr(var, 'shape'):  # numpy array
                            del var
                        del globals()[var_name]
                        to_delete.append(var_name)
                    except:
                        pass
        
        if to_delete:
            print(f"🗑️  Deleted {len(to_delete)} global variables")
    
    # 3. 多次强制垃圾回收
    if force_gc:
        for _ in range(5):  # 增加垃圾回收次数
            collected = gc.collect()
            if collected == 0:
                break
    
    # 4. 强制释放numpy内存
    try:
        import numpy as np
        np.seterr(all='ignore')
    except:
        pass
    
    print("🧹 Memory cleared")

def force_release_resources():
    """强制释放所有资源（包括CPU和内存）"""
    print("🧹 Force releasing all resources...")
    
    # 清理内存
    clear_memory(force_gc=True)
    
    # 检查CPU和内存
    check_cpu_usage()
    check_memory_usage()
    
    # 等待系统稳定
    import time
    time.sleep(1)
    
    # 再次检查
    final_cpu = check_cpu_usage()
    final_mem = check_memory_usage()
    
    if final_cpu > 20:
        print(f"⚠️  CPU still high ({final_cpu:.1f}%), may need manual intervention")
    else:
        print("✅ Resources released successfully")

def load_data(exp):
    # 检查缓存
    if exp in _data_cache:
        return _data_cache[exp]
    
    folder = os.path.join(root_folder_path, exp)
    dt = 1 / 40000

    # 所有可能加载的变量名与对应文件名（不含扩展名）
    npy_files = {
        'v': 'dend_v_array',
        'i': 'dend_i_array',
        'nmda': 'dend_nmda_i_array',
        'ampa': 'dend_ampa_i_array',
        'nmda_g': 'dend_nmda_g_array',
        'ampa_g': 'dend_ampa_g_array',
        'soma': 'soma_v_array',
        'apic_v': 'apic_v_array',
        'apic_ica': 'apic_ica_array',
        'soma_i': 'soma_i_array',
        'trunk_v': 'trunk_v_array',
        'basal_v': 'basal_v_array',
        'tuft_v': 'tuft_v_array',
        'basal_bg_i_nmda': 'basal_bg_i_nmda_array',
        'basal_bg_i_ampa': 'basal_bg_i_ampa_array',
        'tuft_bg_i_nmda': 'tuft_bg_i_nmda_array',
        'tuft_bg_i_ampa': 'tuft_bg_i_ampa_array',
    }

    data = {}

    for var_name, file_base in npy_files.items():
        file_path = os.path.join(folder, f"{file_base}.npy")
        if os.path.exists(file_path):
            data[var_name] = np.load(file_path)
        else:
            data[var_name] = None  # 或者不加入，如果你更喜欢 dict.get(...)

    # 加载 simulation info
    with open(os.path.join(folder, 'simulation_params.json')) as f:
        simu_info = json.load(f)

    # 加载 section_synapse_df.csv
    sec_syn_df_path = os.path.join(folder, 'section_synapse_df.csv')
    if os.path.exists(sec_syn_df_path):
        sec_syn_df = pd.read_csv(sec_syn_df_path)
    else:
        sec_syn_df = None

    data['dt'] = dt
    data['simu_info'] = simu_info
    data['sec_syn_df'] = sec_syn_df

    # 缓存数据
    _data_cache[exp] = data
    return data

def nonlinearity_visualization(exp, data, rec_loc, attr):

    if data is None:
        data = load_data(exp)
    v, soma, apic_v, dt = [data.get(k) for k in ('v', 'soma', 'apic_v', 'dt')]
    simu_info = data['simu_info']  # 必须存在
    
    if v.ndim == 5:
        v = np.mean(v, axis=2) # shape: [num_clusters, num_times, num_affs, num_trials]
        soma = np.mean(soma, axis=1) # shape: [num_times, num_affs, num_trials]
        apic_v = np.mean(apic_v, axis=1) # shape: [num_times, num_affs, num_trials]
        
    t = simu_info['time point of stimulation']
    t_start, t_end = (t-20)*40, (t+100)*40
    x = np.arange(0, t_end-t_start)*dt # for area calculation
    
    # 优化：使用更高效的切片和reshape操作
    v_base_trace = v[:,:,0:1,:]  # 保持维度，避免reshape
    soma_base_trace = soma[:,0:1,:]
    apic_v_base_trace = apic_v[:,0:1,:]
    
    # Initialize variables
    EPSP_array = None
   
    if rec_loc == 'dend':
        dend_delta = np.mean(v[:, t_start:t_end, :, :] - v_base_trace[:, t_start:t_end, :, :], axis=-1)  # shape: [num_clusters, t, num_affs]
        if attr == 'peak':
            EPSP_array = np.mean(np.max(dend_delta, axis=1), axis=0)  # [num_affs]
        elif attr == 'area':
            dend_over_baseline = np.clip(dend_delta, 1, None)  # [num_clus, t, num_affs]
            EPSP_array = np.mean(np.trapz(dend_over_baseline, x, axis=1), axis=0)

    elif rec_loc == 'soma':
        soma_delta = np.mean(soma[t_start:t_end, :, :] - soma_base_trace[t_start:t_end, :, :], axis=-1)  # shape: [t, num_affs]
        if attr == 'peak':
            EPSP_array = np.max(soma_delta, axis=0)  # [num_affs]

            if 'expected' in exp:
                # 优化：向量化expected计算
                max_values = np.max(soma_delta, axis=0)
                EPSP_array = [np.sum(max_values[:1+2*i]) for i in range(37)]

        elif attr == 'area':
            soma_over_baseline = np.clip(soma_delta, 0, None)  # [t, num_affs]
            EPSP_array = np.trapz(soma_over_baseline, x, axis=0)

    elif rec_loc == 'nexus':
        apic_v_delta = np.mean(apic_v[t_start:t_end, :, :] - apic_v_base_trace[t_start:t_end, :, :], axis=-1)  # shape: [t, num_affs]
        if attr == 'peak':
            EPSP_array = np.max(apic_v_delta , axis=0)  # [num_affs]

            if 'expected' in exp:
                # 优化：向量化expected计算
                max_values = np.max(apic_v_delta, axis=0)
                EPSP_array = [np.sum(max_values[:1+2*i]) for i in range(37)]

        elif attr == 'area':
            apic_v_over_baseline = np.clip(apic_v_delta , 0, None)  # [t, num_affs]
            EPSP_array = np.trapz(apic_v_over_baseline, x, axis=0)

    return EPSP_array 

def full_nonlinearity_visualization(exp_list, idx_list, rec_loc_list, attr_list, num_epochs=10):
    ### Load data ###
    exp, idx, rec_loc, attr = exp_list[0], idx_list[0], rec_loc_list[0], attr_list[0]

    data_dict = {}
    for epoch_idx in range(num_epochs):
        epoch_path = exp + '/' + str(idx) + '/' + str(epoch_idx + 1) + '/'
        data_dict[epoch_idx] = load_data(epoch_path)

    EPSP_array_list = []
    
    for epoch_idx in range(num_epochs):
        # Only create plot_attr if plot_flag is True
     
        EPSP_array = nonlinearity_visualization(exp + '/' + str(idx) + '/' + str(epoch_idx + 1) + '/', 
                                                                data_dict[epoch_idx], rec_loc, attr)
        
        EPSP_array_list.append(EPSP_array)

    # 优化：使用numpy数组操作，避免重复转换
    EPSP_matrix = np.array(EPSP_array_list) # shape: [num_epochs, num_affs]
    avg_EPSP_array = np.mean(EPSP_matrix, axis=0)

    return avg_EPSP_array, EPSP_matrix

class GlobalVarManager:
    def __init__(self, name_pattern=None, exclude_vars=None):
        """
        初始化全局变量管理器
        
        Args:
            name_pattern: 变量名匹配的正则表达式模式
            exclude_vars: 要排除的变量名列表
        """
        self.name_pattern = name_pattern
        self.exclude_vars = exclude_vars or [
            '__builtins__', '__cached__', '__doc__', '__file__', 
            '__loader__', '__name__', '__package__', '__spec__',
            'inspect', 'pickle', 'os', 'datetime', 'GlobalVarManager'
        ]
    
    def get_all_globals(self):
        """获取当前所有全局变量"""
        globals_dict = {}
        current_globals = globals()
        
        for var_name, var_value in current_globals.items():
            # 使用正则表达式匹配变量名
            if self.name_pattern and not re.search(self.name_pattern, var_name, re.IGNORECASE):
                continue
            
            # 排除内置变量和指定的排除变量
            if (not var_name.startswith('__') and 
                var_name not in self.exclude_vars and
                not inspect.ismodule(var_value) and
                not inspect.isfunction(var_value) and
                not inspect.isclass(var_value)):
                
                try:
                    pickle.dumps(var_value)
                    globals_dict[var_name] = var_value
                except (pickle.PicklingError, TypeError, AttributeError):
                    print(f"警告: 变量 '{var_name}' 无法序列化，已跳过")
                    continue
        
        return globals_dict
    
    def save_globals(self, filename=None, include_timestamp=True):
        """
        保存所有全局变量到pickle文件
        
        Args:
            filename: 文件名，如果为None则自动生成
            include_timestamp: 是否在文件名中包含时间戳
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"globals_backup_{timestamp}.pkl"
        
        globals_dict = self.get_all_globals()
        
        # 添加元数据
        metadata = {
            'saved_at': datetime.now().isoformat(),
            'variables_count': len(globals_dict),
            'variable_names': list(globals_dict.keys())
        }
        
        save_data = {
            'metadata': metadata,
            'globals': globals_dict
        }
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"✅ 成功保存 {len(globals_dict)} 个全局变量到: {filename}")
            print(f"📋 保存的变量: {list(globals_dict.keys())}")
            return filename
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return None
    
    def load_globals(self, filename, overwrite_existing=True):
        """
        从pickle文件加载全局变量
        
        Args:
            filename: 要加载的文件名
            overwrite_existing: 是否覆盖已存在的变量
        """
        try:
            with open(filename, 'rb') as f:
                save_data = pickle.load(f)
            
            metadata = save_data.get('metadata', {})
            globals_dict = save_data.get('globals', {})
            
            print(f"📂 从文件加载: {filename}")
            print(f"⏰ 保存时间: {metadata.get('saved_at', '未知')}")
            print(f"📊 变量数量: {metadata.get('variables_count', len(globals_dict))}")
            
            loaded_count = 0
            skipped_count = 0
            
            for var_name, var_value in globals_dict.items():
                if var_name in globals() and not overwrite_existing:
                    print(f"⚠️  跳过已存在的变量: {var_name}")
                    skipped_count += 1
                    continue
                
                globals()[var_name] = var_value
                loaded_count += 1
            
            print(f"✅ 成功加载 {loaded_count} 个变量")
            if skipped_count > 0:
                print(f"⚠️  跳过 {skipped_count} 个已存在的变量")
            
            return True
            
        except FileNotFoundError:
            print(f"❌ 文件不存在: {filename}")
            return False
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False
    
    def list_saved_files(self, directory='.'):
        """列出所有保存的全局变量文件"""
        files = [f for f in os.listdir(directory) if f.startswith('globals_backup_') and f.endswith('.pkl')]
        files.sort(reverse=True)  # 最新的文件在前
        
        if not files:
            print("📁 没有找到保存的全局变量文件")
            return []
        
        print("📁 找到以下保存的全局变量文件:")
        for i, file in enumerate(files, 1):
            file_path = os.path.join(directory, file)
            file_size = os.path.getsize(file_path)
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(f"  {i}. {file} ({file_size/1024:.1f} KB, {file_time.strftime('%Y-%m-%d %H:%M:%S')})")
        
        return files

def save_epsps_global():
    """只保存EPSP相关的全局变量"""
    gvm = GlobalVarManager(name_pattern=r'.*EPSP.*')  # 匹配包含EPSP的变量
    
    # 获取所有EPSP相关变量
    epsp_vars = gvm.get_all_globals()
    
    if not epsp_vars:
        print("❌ 没有找到包含'EPSP'的变量")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"epsps_backup_{timestamp}.pkl"
    
    return gvm.save_globals(filename)

def load_epsps_global(filename=None):
    """加载EPSP相关的全局变量"""
    gvm = GlobalVarManager(name_pattern=r'.*EPSP.*')  # 匹配包含EPSP的变量
    
    if filename is None:
        files = [f for f in os.listdir('.') if f.startswith('epsps_backup_') and f.endswith('.pkl')]
        if files:
            filename = sorted(files, reverse=True)[0]
        else:
            print("❌ 没有找到EPSP变量保存文件")
            return False
    
    return gvm.load_globals(filename)

def _single_analysis_task(task_params):
    """
    Wrapper function for a single analysis task (used for parallel processing).
    """
    prefix, filename_template, anal_loc, rec_loc, attr, range_idx, \
        num_epochs, iter_start_idx, iter_end_idx, iter_step = task_params
    
    iter_times = iter_end_idx - iter_start_idx
    
    var_base = f'{prefix}_{anal_loc}_{attr}_{rec_loc}_{range_idx}'
    
    try:
        avg_EPSP_array, EPSP_matrix = full_nonlinearity_visualization(
            [anal_loc + f'_range{range_idx}' + filename_template] * iter_times,
            list(range(iter_start_idx, iter_end_idx, iter_step)),
            [rec_loc] * iter_times, [attr] * iter_times,
            num_epochs=num_epochs
        )
        
        return var_base, avg_EPSP_array, EPSP_matrix
    except Exception as e:
        print(f'❌ Error processing {var_base}: {e}')
        return var_base, None, None

def batch_nonlinearity_analysis(prefix, filename_template, anal_loc_list, rec_loc_list_map, 
                                  num_epochs=50, iter_start_idx=1, iter_end_idx=2, iter_step=1,
                                  parallel=False, max_workers=None):
    """
    Generic function to batch run nonlinearity analysis for different conditions.
    
    Parameters:
    -----------
    prefix : str
        Variable name prefix (e.g., 'vitro_N+A', 'vitro_A_distr')
    filename_template : str  
        Filename template (e.g., '_clus_invitro_singclus', '_distr_invitro_singclus_AMPA')
    anal_loc_list : list
        List of anatomical locations (e.g., ['basal', 'apical'])
    rec_loc_list_map : dict
        Mapping from anal_loc to list of rec_locs (e.g., {'basal': ['dend', 'soma'], 'apical': ['dend', 'nexus']})
    num_epochs : int
        Number of epochs to analyze
    iter_start_idx : int
        Starting index for iterations
    iter_end_idx : int
        Ending index for iterations
    iter_step : int
        Step size for iterations
    parallel : bool
        Whether to use parallel processing (default: True)
    max_workers : int
        Maximum number of worker processes (default: CPU count)
    """
    global _data_cache
    
    # Generate all task parameters
    task_list = []
    for anal_loc in anal_loc_list:
        rec_loc_list = rec_loc_list_map.get(anal_loc, ['dend'])
        
        for attr in ['area']:
            for rec_loc in rec_loc_list:
                for range_idx in [0, 1, 2]:
                    task_params = (
                        prefix, filename_template, anal_loc, rec_loc, attr, range_idx,
                        num_epochs, iter_start_idx, iter_end_idx, iter_step
                    )
                    task_list.append(task_params)
    
    print(f'📊 Starting batch analysis: {len(task_list)} tasks')
    print(f'⚙️  Parallel mode: {parallel}')
    
    # Check initial memory usage
    # clear_memory()
    # check_memory_usage()
    
    # Sequential processing
    print(f'🔄 Sequential mode')
    for i, task in enumerate(task_list, 1):
        var_base, avg_EPSP_array, EPSP_matrix = _single_analysis_task(task)
        
        if EPSP_matrix is not None:
            globals()[f'{var_base}_EPSP_array'] = avg_EPSP_array
            globals()[f'{var_base}_EPSP_matrix'] = EPSP_matrix
            print(f'✓ [{i}/{len(task_list)}] Completed: {var_base}')
        
        # # Periodically clear cache to free memory
        # if i % 5 == 0:
        #     _data_cache.clear()
        #     gc.collect()
    
    print(f'✅ Batch analysis completed!')
    
    # Final resource cleanup
    print('\n🧹 Starting final cleanup...')
    force_release_resources()

#### All conditions in one call:

# Complete example for all vitro conditions
iter_start_idx, iter_end_idx = 1, 2
iter_step, num_epochs = 1, 50

# 1. Vitro N+A Clustered & Distributed (SAFE MODE - Sequential)
print("🚨 Running in SAFE MODE to prevent system restart")
print("💡 To enable parallel processing, set parallel=True (use with caution)")

batch_nonlinearity_analysis('vitro_N+A', '_clus_invitro_singclus', ['basal', 'apical'], 
                            {'basal': ['dend', 'soma'], 'apical': ['dend', 'soma']}, num_epochs)

batch_nonlinearity_analysis('vitro_N+A_distr', '_distr_invitro_singclus', ['basal', 'apical'],
                            {'basal': ['dend', 'soma'], 'apical': ['dend', 'soma']}, num_epochs)

# # 4. Vivo N+A Clustered & Distributed
# batch_nonlinearity_analysis('vivo_N+A', '_clus_invivo_singclus', ['basal', 'apical'],
#                             {'basal': ['dend', 'soma'], 'apical': ['dend', 'nexus']}, num_epochs)

# batch_nonlinearity_analysis('vivo_N+A_distr', '_distr_invivo_singclus', ['basal', 'apical'],
#                             {'basal': ['dend', 'soma'], 'apical': ['dend', 'nexus']}, num_epochs)


# # 2. Vitro AMPA Clustered & Distributed
# batch_nonlinearity_analysis('vitro_A', '_clus_invitro_singclus_AMPA', ['basal', 'apical'],
#                             {'basal': ['dend', 'soma'], 'apical': ['dend', 'nexus']}, num_epochs)

# batch_nonlinearity_analysis('vitro_A_distr', '_distr_invitro_singclus_AMPA', ['basal', 'apical'],
#                             {'basal': ['dend', 'soma'], 'apical': ['dend', 'nexus']}, num_epochs)

# # 3. Vitro N/A 3:1 Clustered & Distributed
# batch_nonlinearity_analysis('vitro_N/A_3', '_clus_invitro_singclus_ratio3', ['basal', 'apical'],
#                             {'basal': ['dend', 'soma'], 'apical': ['dend', 'nexus']}, num_epochs)

# batch_nonlinearity_analysis('vitro_N/A_3_distr', '_distr_invitro_singclus_ratio3', ['basal', 'apical'],
#                             {'basal': ['dend', 'soma'], 'apical': ['dend', 'nexus']}, num_epochs)



# # 5. Vivo AMPA Clustered & Distributed
# batch_nonlinearity_analysis('vivo_A', '_clus_invivo_singclus_AMPA', ['basal', 'apical'],
#                             {'basal': ['dend', 'soma'], 'apical': ['dend', 'nexus']}, num_epochs)

# batch_nonlinearity_analysis('vivo_A_distr', '_distr_invivo_singclus_AMPA', ['basal', 'apical'],
#                             {'basal': ['dend', 'soma'], 'apical': ['dend', 'nexus']}, num_epochs)

# root_folder_path = '/G/results/simulation_singclus_Oct25'

# # 6. Vivo N+A 10ms Clustered & Distributed
# for time_range in [10, 20, 40]:
#     batch_nonlinearity_analysis(f'vivo_N+A_{time_range}ms', f'_clus_invivo_singclus_t{time_range}ms', ['basal', 'apical'],
#                                 {'basal': ['soma'], 'apical': ['nexus']}, num_epochs)
#     batch_nonlinearity_analysis(f'vivo_N+A_distr_{time_range}ms', f'_distr_invivo_singclus_t{time_range}ms', ['basal', 'apical'],
#                                 {'basal': ['soma'], 'apical': ['nexus']}, num_epochs)

saved_file = save_epsps_global()