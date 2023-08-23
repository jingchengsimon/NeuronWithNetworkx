from neuron import h
h.nrn_load_dll('./mod/nrnmech.dll')

# spt = h.Vector(10).indgen(10, 2)
spt = h.Vector([1,3,5])
vs = h.VecStim()
vs.play(spt)

def pr():
  print (h.t)

nc = h.NetCon(vs, None)
nc.record(pr)

cvode = h.CVode()
h.finitialize()
cvode.solve(20)