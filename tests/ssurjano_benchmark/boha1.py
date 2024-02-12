import numpy as np
def boha1(xx):
    ##########################################################################
    #BOHACHEVSKY FUNCTION 1
    ############################
    
    x1, x2 = xx[:-1], xx[-1]
    term1 = x1**2
    term2 = 2*x2**2
    term3 = -0.3 * np.cos(3*np.pi*x1)
    term4 = -0.4 * np.cos(4*np.pi*x2)
    
    y = term1 + term2 + term3 + term4 + 0.7
    
    return(y)