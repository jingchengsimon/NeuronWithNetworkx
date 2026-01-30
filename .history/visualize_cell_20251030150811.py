import matplotlib.pyplot as plt
from neurom import load_morphology

# 加载 SWC 文件
morpho = load_morphology('L5b_neuron.swc')

# 绘制二维投影
fig, ax = plt.subplots()
for neurite in morpho.neurites:
    points = neurite.points
    ax.plot(points[:,0], points[:,1], color='black')  # XY 投影

# 保存为矢量图
fig.savefig('L5b_neuron.svg')  # 或者 .pdf
