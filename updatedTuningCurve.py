import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 假设有八个角度和每个角度对应的15个spike个数的值
angles = np.arange(0, 360, 45)
spike_counts = np.random.randint(10, 20, size=(8, 15))  # 示例中使用随机生成的数据

file_path = './num_spikes_df.csv'
spike_counts = np.array(pd.read_csv(file_path, index_col=None, header=0).T)
# 计算每个角度的平均值和标准差
mean_spike_counts = np.mean(spike_counts, axis=1)
std_dev_spike_counts = np.std(spike_counts, axis=1)

# 绘制调谐曲线
plt.errorbar(angles, mean_spike_counts, yerr=std_dev_spike_counts, fmt='o-', capsize=5, label='Tuning Curve')

# 在图上标出15个值对应的散点
for angle, spike_count_values in zip(angles, spike_counts):
    plt.scatter([angle] * len(spike_count_values), spike_count_values, color='gray', alpha=0.5)

# 添加标签和标题
plt.xlabel('Angle (degrees)')
plt.ylabel('Average Spike Count')
plt.title('Tuning Curve with Scatter Points')
plt.legend()

# 显示图形
plt.show()
