from neuron import h

def recur_dist_to_soma(sec, loc, initial=True): 
        
    if h.SectionRef(sec).has_parent():
        sec_len = sec.L * loc if initial else sec.L
        parent_sec = h.SectionRef(sec).parent
        return sec_len + recur_dist_to_soma(parent_sec, loc, initial=False)
    else:
        # If there is no parent, the section is the soma, return half lenth of the soma
        return sec.L * 0.5 
    
def recur_dist_to_root(sec, loc, root, initial=True):