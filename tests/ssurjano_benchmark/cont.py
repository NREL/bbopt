import numpy as np
def cont(xx, u=np.array([0.5]*len(xx)), a=np.array([5]*len(xx))):
    ########################################################################
    # CONTINUOUS INTEGRAND FAMILY
    #####################################################################
    
    xx = np.array(xx)
    u = np.array(u, dtype=np.float32)
    a = np.array(a, dtype=np.float32)
    
    sum = np.sum(a * np.abs(xx-u))
    
    y = np.exp(-sum)
    
    return(y)