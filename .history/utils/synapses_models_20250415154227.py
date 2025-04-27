from neuron import h
import numpy as np

def AMPANMDA(initW, sec_x, sec_id, basal_channel_type):
        """Create a bg2pyr synapse
        :param syn_params: parameters of a synapse
        :param sec_x: normalized distance along the section
        :param sec_id: target section
        :return: NEURON synapse object
        """

        # np.random.seed(42)  # Set the random seed for reproducibility

        # def lognormal(m, s):
        #     mean = np.log(m) - 0.5 * np.log((s/m)**2+1)
        #     std = np.sqrt(np.log((s/m)**2 + 1))
        #     #import pdb; pdb.set_trace()
        #     return max(np.random.lognormal(mean, std, 1), 0.00000001)

        # pyrWeight_m = 0.45#0.229#0.24575#0.95
        # pyrWeight_s = 0.345#1.3

        if basal_channel_type == 'AMPANMDA':
            lsyn = h.ProbAMPANMDA2(sec_x, sec=sec_id)
        elif basal_channel_type == 'AMPA':
            lsyn = h.ProbAMPA(sec_x, sec=sec_id)

        # if syn_params.get('tau_r_AMPA'):
        #     lsyn.tau_r_AMPA = float(syn_params['tau_r_AMPA'])
        # if syn_params.get('tau_d_AMPA'):
        #     lsyn.tau_d_AMPA = float(syn_params['tau_d_AMPA'])

        # if syn_params.get('tau_r_NMDA'):
        #     lsyn.tau_r_NMDA = float(syn_params['tau_r_NMDA'])
        # if syn_params.get('tau_d_NMDA'):
        #     lsyn.tau_d_NMDA = float(syn_params['tau_d_NMDA'])
        # if syn_params.get('Use'):
        #     lsyn.Use = float(syn_params['Use'])
        # if syn_params.get('Dep'):
        #     lsyn.Dep = float(syn_params['Dep'])
        # if syn_params.get('Fac'):
        #     lsyn.Fac = float(syn_params['Fac'])
        # if syn_params.get('e'):
        #     lsyn.e = float(syn_params['e'])

        # if syn_params.get('initW'):
        lsyn.initW = float(initW)

            # h.distance(sec=sec_id.cell().soma[0])
            # dist = h.distance(sec_id(sec_x))
            # fullsecname = sec_id.name()
            # sec_type = fullsecname.split(".")[1][:4]
            # sec_id = int(fullsecname.split("[")[-1].split("]")[0])

            # dend = lambda x: ( 1.001 ** x )
            # close_apic = lambda x: ( 1.002 ** x )
            # #far_apic = lambda x: ( 1.002 ** x )
            # far_apic = lambda x: 1

            # if sec_type == "dend":
            #     base = float(np.clip(lognormal(pyrWeight_m, pyrWeight_s), 0, 5))
            #     lsyn.initW = base * dend(dist)
            # elif sec_type == "apic":
            #     if dist < 750:
            #         base = float(np.clip(lognormal(pyrWeight_m, pyrWeight_s), 0, 5))
            #         lsyn.initW = base * close_apic(dist)
            #     else:
            #         base = float(np.clip(lognormal(0.17, 0.2), 0, 5))
            #         lsyn.initW = base * far_apic(dist)

            # lsyn.initW = np.clip(float(lsyn.initW), 0, 5)

        # if syn_params.get('u0'):
        #     lsyn.u0 = float(syn_params['u0'])

        return lsyn
