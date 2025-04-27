TITLE AMPA receptor with presynaptic short-term plasticity 


COMMENT
AMPA receptor conductance using a dual-exponential profile
presynaptic short-term plasticity based on Fuhrmann et al. 2002
Implemented by Srikanth Ramaswamy, Blue Brain Project, July 2009
Etay: changed weight to be equal for AMPA, gmax accessible in Neuron

ENDCOMMENT


NEURON {

        POINT_PROCESS ProbAMPA 
        RANGE tau_r_AMPA, tau_d_AMPA, tau_r_NMDA, tau_d_NMDA
        RANGE Use, u, Dep, Fac, u0
        RANGE i, i_AMPA, g_AMPA, e, initW
        NONSPECIFIC_CURRENT i_AMPA
	POINTER rng
}

PARAMETER {

        tau_r_AMPA = 1 (ms) :0.2   (ms)  : dual-exponential conductance profile
        tau_d_AMPA = 2 (ms) :1.7    (ms)  : IMPORTANT: tau_r < tau_d
        tau_r_NMDA = 5 (ms) :0.29   (ms) : dual-exponential conductance profile
        tau_d_NMDA = 90 (ms) :43     (ms) : IMPORTANT: tau_r < tau_d
        Use = 1.0   (1)   : Utilization of synaptic efficacy (just initial values! Use, Dep and Fac are overwritten by BlueBuilder assigned values) 
        Dep = 100   (ms)  : relaxation time constant from depression
        Fac = 10   (ms)  :  relaxation time constant from facilitation
        e = 0     (mV)  : AMPA reversal potential
	mg = 1   (mM)  : initial concentration of mg2+
        mggate
    	initW = .001 (uS)
        ratio_NMDA_to_AMPA = 1.1 : 1.1
    	u0 = 0 :initial value of u, which is the running value of Use
}

COMMENT
The Verbatim block is needed to generate random nos. from a uniform distribution between 0 and 1 
for comparison with Pr to decide whether to activate the synapse or not
ENDCOMMENT
   
VERBATIM

#include<stdlib.h>
#include<stdio.h>
#include<math.h>

double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);

ENDVERBATIM
  

ASSIGNED {

        v (mV)
        i (nA)
	i_AMPA (nA)
        g_AMPA (uS)
        factor_AMPA
	rng
}

STATE {

        A_AMPA       : AMPA state variable to construct the dual-exponential profile - decays with conductance tau_r_AMPA
        B_AMPA       : AMPA state variable to construct the dual-exponential profile - decays with conductance tau_d_AMPA	
}

INITIAL{

        LOCAL tp_AMPA
        
	A_AMPA = 0
        B_AMPA = 0
        
	tp_AMPA = (tau_r_AMPA*tau_d_AMPA)/(tau_d_AMPA-tau_r_AMPA)*log(tau_d_AMPA/tau_r_AMPA) :time to peak of the conductance

	factor_AMPA = -exp(-tp_AMPA/tau_r_AMPA)+exp(-tp_AMPA/tau_d_AMPA) :AMPA Normalization factor - so that when t = tp_AMPA, gsyn = gpeak
        factor_AMPA = 1/factor_AMPA	
}

BREAKPOINT {

        SOLVE state METHOD cnexp
	g_AMPA = initW*(B_AMPA-A_AMPA) / ratio_NMDA_to_AMPA :compute time varying conductance as the difference of state variables B_AMPA and A_AMPA
	i_AMPA = g_AMPA*(v-e) :compute the AMPA driving force based on the time varying conductance, membrane potential, and AMPA reversal
	i = i_AMPA
}

DERIVATIVE state{

        A_AMPA' = -A_AMPA/tau_r_AMPA
        B_AMPA' = -B_AMPA/tau_d_AMPA
}


NET_RECEIVE (weight,weight_AMPA, Pv, Pr, u, tsyn (ms)){
	
	weight_AMPA = weight
        A_AMPA = A_AMPA + weight_AMPA*factor_AMPA
        B_AMPA = B_AMPA + weight_AMPA*factor_AMPA

	
}

