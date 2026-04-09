from neuron import gui, h

h.load_file("import3d.hoc") 
h.nrn_load_dll('./mod/nrnmech.dll')
h.load_file('./modelFile/L5PCbiophys3.hoc')
h.load_file('./modelFile/L5PCtemplate.hoc')

complex_cell = h.L5PCtemplate('./modelFile/cell1.asc')

s = h.Shape()
s.show(0)