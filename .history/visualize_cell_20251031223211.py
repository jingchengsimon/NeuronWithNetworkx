from neuron import h, gui
import matplotlib.pyplot as plt

# 1. 导入 SWC
# cell = h.Import3d_SWC_read()
swc_file = './modelFile/cell1.asc'


h.load_file("import3d.hoc")

h.nrn_load_dll('./mod/x86_64/.libs/libnrnmech.so') # For Linux/Mac
h.load_file('./modelFile/L5PCbiophys3.hoc')
h.load_file('./modelFile/L5PCtemplate.hoc')

complex_cell = h.L5PCtemplate(swc_file)


s = h.PlotShape(False)
s.plot(plt)

plt.savefig("morphology.pdf", bbox_inches='tight')
