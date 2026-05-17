#reduction of L5_PC using Neuron_Reduce

from __future__ import division
import os
import logging
from neuron import gui, h
import numpy as np
import time
import matplotlib.pyplot as plt
# Import from local subtree_reductor_func.py instead of neuron_reduce package
from subtree_reductor_func import subtree_reductor

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))



#Create a L5_PC model
h.load_file('./modelFile/L5PCbiophys3.hoc')
h.load_file("import3d.hoc")
h.load_file('./modelFile/L5PCtemplate.hoc')
complex_cell = h.L5PCtemplate('./modelFile/cell1.asc')
h.celsius = 37
h.v_init = complex_cell.soma[0].e_pas


#Add synapses to the model
synapses_list, netstims_list, netcons_list, randoms_list = [], [], [] ,[]

all_segments = [i for j in map(list,list(complex_cell.apical)) for i in j] + [i for j in map(list,list(complex_cell.basal)) for i in j]
len_per_segment = np.array([seg.sec.L/seg.sec.nseg for seg in all_segments])
rnd = np.random.RandomState(10)
for i in range(10000):
    seg_for_synapse = rnd.choice(all_segments,   p=len_per_segment/sum(len_per_segment))
    synapses_list.append(h.Exp2Syn(seg_for_synapse))
    if rnd.uniform()<0.85:
        e_syn, tau1, tau2, spike_interval, syn_weight = 0, 0.3, 1.8,  1000/2.5, 0.0016
    else:
        e_syn, tau1, tau2, spike_interval, syn_weight = -86, 1,   8,   1000/15.0, 0.0008


    synapses_list[i].e, synapses_list[i].tau1, synapses_list[i].tau2 = e_syn, tau1, tau2

    netstims_list.append(h.NetStim())
    netstims_list[i].interval, netstims_list[i].number, netstims_list[i].start, netstims_list[i].noise = spike_interval, 9e9, 100, 1

    randoms_list.append(h.Random())
    randoms_list[i].Random123(i)
    randoms_list[i].negexp(1)
    netstims_list[i].noiseFromRandom(randoms_list[i])

    netcons_list.append(h.NetCon(netstims_list[i], synapses_list[i] ))
    netcons_list[i].delay, netcons_list[i].weight[0] = 0, syn_weight

#Simulate the full neuron for 1 seconds
soma_v = h.Vector()
soma_v.record(complex_cell.soma[0](0.5)._ref_v)

time_v = h.Vector()
time_v.record(h._ref_t)


# #apply Neuron_Reduce to simplify the cell
reduced_cell, synapses_list, netcons_list, new_cables_nsegs = subtree_reductor(complex_cell, synapses_list, netcons_list, reduction_frequency=0)
for r in randoms_list:r.seq(1) #reset random

# Print the number of segments in new cables
print(f"Number of segments in new cables: {new_cables_nsegs}")
print(f"Total number of segments: {sum(new_cables_nsegs)}")

# Record voltage from first 5 new segments
all_new_segments = []
if reduced_cell.apic is not None:
    all_new_segments.extend(list(reduced_cell.apic))
for dend_sec in reduced_cell.dend:
    all_new_segments.extend(list(dend_sec))

# Record voltage from first 5 segments
new_seg_v_list = []
for i, seg in enumerate(all_new_segments[:5]):
    v_vec = h.Vector()
    v_vec.record(seg._ref_v)
    new_seg_v_list.append(v_vec)
    print(f"Recording voltage from segment {i+1}: {seg}")

print(len(synapses_list))
print(len(netcons_list))
#Running the simulation again but now on the reduced cell
st = time.time()
h.run()
print('reduced cell simulation time {:.4f}'.format(time.time()-st))
reduced_cell_v = list(soma_v)

#plotting the results
plt.figure()
plt.plot(time_v, reduced_cell_v,  label='reduced cell soma')
# Plot voltage from first 5 new segments
for i, v_vec in enumerate(new_seg_v_list):
    plt.plot(time_v, list(v_vec), label=f'new segment {i+1}')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.legend()
plt.show()
