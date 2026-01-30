from neuron import h, gui
import matplotlib.pyplot as plt

# 1. 导入 SWC
# cell = h.Import3d_SWC_read()
# cell.input('./modelFile/cell1.asc')
# i3d = h.Import3d_GUI(cell, 0)
# i3d.instantiate(None)

h.load_file('stdrun.hoc')
h.load_file('import3d.hoc')
h.load_file('import3d_gui.hoc')
h.load_File('./modelFile/cell1.asc')

# cell = h.Import3d_SWC_read()
# cell.input('./modelFile/cell1.asc')
# i3d = h.Import3d_GUI(cell, 0)
# i3d.instantiate(None)

# 2. 绘制
ps = h.PlotShape(False)
ps.variable('none')
ps.show(0)
ps.scale(10)

# 3. 着色
for sec in h.allsec():
    if 'soma' in sec.name():
        ps.color(2, sec=sec)
    elif 'apic' in sec.name():
        ps.color(3, sec=sec)
    elif 'dend' in sec.name():
        ps.color(4, sec=sec)

# 4. 保存为矢量图
# plt.savefig("morphology.svg", bbox_inches='tight')
plt.savefig("morphology.pdf", bbox_inches='tight')
plt.show()
