#reduction of L5_PC using Neuron_Reduce

from __future__ import division
import os
import logging
from neuron import gui,h
import numpy as np
# import neuron_reduce
import time
import json
import matplotlib.pyplot as plt
from utils.synapses_models import AMPANMDA

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

#Create a L5_PC model
swc_file = './modelFile/cell1.asc'
h.load_file("import3d.hoc")
        
# changing the loc of folder seems to make loading the relative path wrong
current_directory = os.path.dirname(__file__)  # 如果在脚本中使用，__file__是指当前脚本的文件名
os.chdir(current_directory)
relative_path = './mod/nrnmech.dll'
nrnmech_path = os.path.abspath(relative_path)
h.nrn_load_dll(nrnmech_path)

h.load_file('./modelFile/L5PCbiophys3.hoc')
h.load_file('./modelFile/L5PCtemplate.hoc')
complex_cell = h.L5PCtemplate(swc_file)
h.celsius = 37

# h.load_file('L5PCbiophys3.hoc')
# h.load_file("import3d.hoc")
# h.load_file('L5PCtemplate.hoc')
# complex_cell = h.L5PCtemplate('cell1.asc')
# h.celsius = 37
# h.v_init = complex_cell.soma[0].e_pas


#Add synapses to the model
synapses_list, netstims_list, netcons_list, randoms_list = [], [], [] ,[]

all_segments = [i for j in map(list,list(complex_cell.apical)) for i in j] + [i for j in map(list,list(complex_cell.basal)) for i in j]
len_per_segment = np.array([seg.sec.L/seg.sec.nseg for seg in all_segments])
rnd = np.random.RandomState(10)
syn_params = json.load(open('./modelFile/AMPANMDA.json', 'r'))
for i in range(1):
    seg_for_synapse = rnd.choice(all_segments,   p=len_per_segment/sum(len_per_segment))
    loc = seg
    synapses_list.append(AMPANMDA(syn_params, section['loc'], section['section_synapse'], 'AMPANMDA')
    # synapses_list.append(h.Exp2Syn(seg_for_synapse))
    e_syn, tau1, tau2, spike_interval, syn_weight = 0, 0.3, 1.8,  1000/2.5, 0.0016

    # if rnd.uniform()<0.85:
    #     e_syn, tau1, tau2, spike_interval, syn_weight = 0, 0.3, 1.8,  1000/2.5, 0.0016
    # else:
    #     e_syn, tau1, tau2, spike_interval, syn_weight = -86, 1,   8,   1000/15.0, 0.0008


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

h.tstop = 1000
st = time.time()
h.run()
print('complex cell simulation time {:.4f}'.format(time.time()-st))
complex_cell_v = list(soma_v)


