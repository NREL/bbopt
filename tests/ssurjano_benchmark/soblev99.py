import numpy as np
def soblev99(xx, b=None, c0=0):
    ##########################################################################
    #
    # SOBOL' & LEVITAN (1999) FUNCTION
    #
    ####################################################################
    
    d = len(xx)
    
    if b is None:
        if d <= 20:
            b = np.array([2, 1.95, 1.9, 1.85, 1.8, 1.75, 1.7, 1.65, 0.4228, 0.3077, 0.2169, 0.1471, 0.0951, 0.0577, 0.0323, 0.0161, 0.0068, 0.0021, 0.0004, 0])
        else:
            raise ValueError('Value of the d-dimensional vector b is required.')
    
    Id = np.ones(d)
    for ii in range(1, d+1):
        bi = b[ii]
        new = (np.exp(bi)-1)/bi
        Id *= new
        
    sum = 0
    for ii in range(1, d+1):
        bi = b[ii]
        xi = xx[ii]
        sum += bi*xi
    
    y = np.exp(sum) - Id + c0
    return y