from neuron import h
import numpy as np

def AMPANMDA(syn_params, sec_x, sec_id, basal_channel_type):
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

        if syn_params.get('initW'):
            lsyn.initW = float(syn_params['initW'])

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

        if syn_params.get('u0'):
            lsyn.u0 = float(syn_params['u0'])
        return lsyn

def Pyr2Pyr(syn_params, sec_x, sec_id):
    """Create a pyr2pyr synapse
    :param syn_params: parameters of a synapse
    :param sec_x: normalized distance along the section
    :param sec_id: target section
    :return: NEURON synapse object
    """
    def lognormal(m, s):
        mean = np.log(m) - 0.5 * np.log((s/m)**2+1)
        std = np.sqrt(np.log((s/m)**2 + 1))
        #import pdb; pdb.set_trace()
        return max(np.random.lognormal(mean, std, 1), 0.00000001)

    pyrWeight_m = 0.45#0.229#0.24575#0.95
    pyrWeight_s = 0.345#1.3

    lsyn = h.pyr2pyr(sec_x, sec=sec_id)

    #Assigns random generator of release probability.
    r = h.Random()
    r.MCellRan4()
    r.uniform(0,1)
    lsyn.setRandObjRef(r)

    lsyn.P_0 = 0.6#np.clip(np.random.normal(0.53, 0.22), 0, 1)#Release probability

    if syn_params.get('AlphaTmax_ampa'):
        lsyn.AlphaTmax_ampa = float(syn_params['AlphaTmax_ampa']) # par.x(21)
    if syn_params.get('Beta_ampa'):
        lsyn.Beta_ampa = float(syn_params['Beta_ampa']) # par.x(22)
    if syn_params.get('Cdur_ampa'):
        lsyn.Cdur_ampa = float(syn_params['Cdur_ampa']) # par.x(23)
    if syn_params.get('gbar_ampa'):
        lsyn.gbar_ampa = float(syn_params['gbar_ampa']) # par.x(24)
    if syn_params.get('Erev_ampa'):
        lsyn.Erev_ampa = float(syn_params['Erev_ampa']) # par.x(16)

    if syn_params.get('AlphaTmax_nmda'):
        lsyn.AlphaTmax_nmda = float(syn_params['AlphaTmax_nmda']) # par.x(25)
    if syn_params.get('Beta_nmda'):
        lsyn.Beta_nmda = float(syn_params['Beta_nmda']) # par.x(26)
    if syn_params.get('Cdur_nmda'):
        lsyn.Cdur_nmda = float(syn_params['Cdur_nmda']) # par.x(27)
    if syn_params.get('gbar_nmda'):
        lsyn.gbar_nmda = float(syn_params['gbar_nmda']) # par.x(28)
    if syn_params.get('Erev_nmda'):
        lsyn.Erev_nmda = float(syn_params['Erev_nmda']) # par.x(16)
    
    if syn_params.get('initW'):
        h.distance(sec=sec_id.cell().soma[0])
        dist = h.distance(sec_id(sec_x))
        fullsecname = sec_id.name()
        sec_type = fullsecname.split(".")[1][:4]
        sec_id = int(fullsecname.split("[")[-1].split("]")[0])

        dend = lambda x: ( 1.00 ** x )
        close_apic = lambda x: ( 1.00 ** x )
        #far_apic = lambda x: ( 1.002 ** x )
        far_apic = lambda x: 1

        if sec_type == "dend":
            base = float(np.clip(lognormal(pyrWeight_m, pyrWeight_s), 0, 5))
            lsyn.initW = base * dend(dist)
        elif sec_type == "apic":
            if dist < 750:
                base = float(np.clip(lognormal(pyrWeight_m, pyrWeight_s), 0, 5))
                lsyn.initW = base * close_apic(dist)
            else:
                base = float(np.clip(lognormal(pyrWeight_m, pyrWeight_s), 0, 5))
                lsyn.initW = base * far_apic(dist)

        lsyn.initW = np.clip(float(lsyn.initW), 0, 5)


    if syn_params.get('Wmax'):
        lsyn.Wmax = float(syn_params['Wmax']) * lsyn.initW # par.x(1) * lsyn.initW
    if syn_params.get('Wmin'):
        lsyn.Wmin = float(syn_params['Wmin']) * lsyn.initW # par.x(2) * lsyn.initW
    #delay = float(syn_params['initW']) # par.x(3) + delayDistance
    #lcon = new NetCon(&v(0.5), lsyn, 0, delay, 1)

    if syn_params.get('lambda1'):
        lsyn.lambda1 = float(syn_params['lambda1']) # par.x(6)
    if syn_params.get('lambda2'):
        lsyn.lambda2 = float(syn_params['lambda2']) # par.x(7)
    if syn_params.get('threshold1'):
        lsyn.threshold1 = float(syn_params['threshold1']) # par.x(8)
    if syn_params.get('threshold2'):
        lsyn.threshold2 = float(syn_params['threshold2']) # par.x(9)
    if syn_params.get('tauD1'):
        lsyn.tauD1 = float(syn_params['tauD1']) # par.x(10)
    if syn_params.get('d1'):
        lsyn.d1 = float(syn_params['d1']) # par.x(11)
    if syn_params.get('tauD2'):
        lsyn.tauD2 = float(syn_params['tauD2']) # par.x(12)
    if syn_params.get('d2'):
        lsyn.d2 = float(syn_params['d2']) # par.x(13)
    if syn_params.get('tauF'):
        lsyn.tauF = float(syn_params['tauF']) # par.x(14)
    if syn_params.get('f'):
        lsyn.f = float(syn_params['f']) # par.x(15)

    if syn_params.get('bACH'):
        lsyn.bACH = float(syn_params['bACH']) # par.x(17)
    if syn_params.get('aDA'):
        lsyn.aDA = float(syn_params['aDA']) # par.x(18)
    if syn_params.get('bDA'):
        lsyn.bDA = float(syn_params['bDA']) # par.x(19)
    if syn_params.get('wACH'):
        lsyn.wACH = float(syn_params['wACH']) # par.x(20)
    
    return lsyn
