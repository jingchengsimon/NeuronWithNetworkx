from neuron import h
import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

# 获取当前脚本的目录
current_directory = os.path.dirname(__file__)  # 如果在脚本中使用，__file__是指当前脚本的文件名
os.chdir(current_directory)

relative_path = './mod/nrnmech.dll'
absolute_path = os.path.join(current_directory, relative_path)
h.nrn_load_dll(absolute_path)

# 定义文件路径
spt_path = 'C:/Users/Windows/Desktop/MIMOlab/Codes/AllenAtlas/results/spt_df/'

# Use glob.glob to extract the csv files with the orientation wanted
spt_file = glob.glob(spt_path + '*_45.0*.csv') # update to os.path.join

# 8 times 15 times 19 (3 connetions randomly to 5 clusters)
# Currently we only have 1 file

# for file_path in spt_file:
#     spt_df = pd.read_csv(file_path, index_col=None, header=0)
    
#     unit_ids = np.sort(spt_df['unit_id'].unique())
#     stimulus_presentation_id = np.sort(spt_df['stimulus_presentation_id'].unique())[0]

#     spt_grouped_df = spt_df.groupby(['unit_id', 'stimulus_presentation_id'])

#     for unit_id in unit_ids:
#       spt_unit = spt_grouped_df.get_group((unit_id, stimulus_presentation_id))
#       spt = h.Vector(spt_unit['spike_time'].values - spt_unit['spike_time'].values[0])
#       vecstim = h.VecStim()
#       vecstim.play(spt)

spt_unit_lists = []
        
for file_path in spt_file:
    spt_df = pd.read_csv(file_path, index_col=None, header=0)
    spt_grouped_df = spt_df.groupby(['unit_id', 'stimulus_presentation_id'])

    unit_ids = np.sort(spt_df['unit_id'].unique())
    stimulus_presentation_id = np.sort(spt_df['stimulus_presentation_id'].unique())[0]

    for unit_id in unit_ids:
        spt_unit = spt_grouped_df.get_group((unit_id, stimulus_presentation_id))
        spt_unit = spt_unit['spike_time'].values - spt_unit['spike_time'].values[0]
        spt_unit_lists.append(spt_unit)

k = 5
results = []  # 用于存储生成的列表
indices = []
rnd = np.random.RandomState(10)
for _ in range(len(unit_ids)):
    sampled = rnd.choice(k, 3)  # 从范围中选择三个不同的整数
    results.append(sampled)

# 查找包含从0到k-1的列表的索引
for i in range(k):
    index_list = [j for j, lst in enumerate(results) if i in lst]
    indices.append(index_list)

vecstim_lists = []
for indices_list in indices:
    spt_unit_summed_list = []
    for index in indices_list:
        spt_unit_summed_list.extend(spt_unit_lists[index])
    spt_unit_summed_list.sort()
    spt_unit_summed_vector = h.Vector(spt_unit_summed_list)
    vecstim = h.VecStim()
    vecstim.play(spt_unit_summed_vector)
    vecstim_lists.append(vecstim)

# def pr():
  # print (h.t)

netcons_list = []
for i in range(50):
  # spt_unit_summed_list = rnd.choice(vecstim_lists)
  # spt_unit_summed_vector = h.Vector(spt_unit_summed_list)
  # vecstim = h.VecStim()
  # vecstim.play(spt_unit_summed_vector)
  vecstim = rnd.choice(vecstim_lists)
  nc = h.NetCon(vecstim, None)
  netcons_list.append(nc)

spike_times = [h.Vector() for _ in netcons_list]
for nc, spike_times_vec in zip(netcons_list, spike_times):
    nc.record(spike_times_vec)

# h.tstop = 2000
h.run()

plt.figure(figsize=(5, 5))
for i, spike_times_vec in enumerate(spike_times):
    try:
        if len(spike_times_vec) > 0:
            print(i)
            plt.vlines(spike_times_vec, i + 0.5, i + 1.5)
    except IndexError:
        continue
      
plt.show()

# nc.record(pr)

# cvode = h.CVode()
# h.finitialize()
# cvode.solve(20)