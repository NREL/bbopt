import numpy as np
def oscill(xx, u=np.array([0.5]), a=np.array([5])):
    ##########################################################################
    #OSCILLATORY INTEGRAND FAMILY
    ############################
    
    term1 = 2*np.pi*u[0]
    sum = np.sum(a*xx)
    y = np.cos(term1 + sum)
    
    return(y)