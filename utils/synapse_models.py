from neuron import h

def AMPANMDA(syn_params, sec_x, sec_id, basal_channel_type):
        """Create a bg2pyr synapse
        :param syn_params: parameters of a synapse
        :param sec_x: normalized distance along the section
        :param sec_id: target section
        :return: NEURON synapse object
        """

        if basal_channel_type == 'AMPANMDA':
            lsyn = h.ProbAMPANMDA2(sec_x, sec=sec_id)
        elif basal_channel_type == 'AMPA':
            lsyn = h.ProbAMPA(sec_x, sec=sec_id)

        if syn_params.get('tau_r_AMPA'):
            lsyn.tau_r_AMPA = float(syn_params['tau_r_AMPA'])
        if syn_params.get('tau_d_AMPA'):
            lsyn.tau_d_AMPA = float(syn_params['tau_d_AMPA'])

        if syn_params.get('tau_r_NMDA'):
            lsyn.tau_r_NMDA = float(syn_params['tau_r_NMDA'])
        if syn_params.get('tau_d_NMDA'):
            lsyn.tau_d_NMDA = float(syn_params['tau_d_NMDA'])
        if syn_params.get('Use'):
            lsyn.Use = float(syn_params['Use'])
        if syn_params.get('Dep'):
            lsyn.Dep = float(syn_params['Dep'])
        if syn_params.get('Fac'):
            lsyn.Fac = float(syn_params['Fac'])
        if syn_params.get('e'):
            lsyn.e = float(syn_params['e'])
        if syn_params.get('initW'):
            lsyn.initW = float(syn_params['initW'])
        if syn_params.get('ratio_NMDA_to_AMPA'):
            lsyn.ratio_NMDA_to_AMPA = float(syn_params['ratio_NMDA_to_AMPA'])
        if syn_params.get('u0'):
            lsyn.u0 = float(syn_params['u0'])

        return lsyn
