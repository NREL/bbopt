import numpy as np
def robot(xx):
    ##########################################################################
    #
    # ROBTEN INPUTS AND OUTPUTS FROM ARGUMENT xx
    #
    ####################################################################
    
    theta = xx[1:4]
    L = xx[5:8]
    
    thetamat = np.tile(theta, (4, 4))
    thetamatlow = thetamat[:-1, :-1]
    thetamatlow[np.triu_indices(thetamatlow.shape)] = 0
    sumtheta = np.sum(np.abs(np.diag(thetamatlow)), axis=1)
    
    u = np.dot(L, np.cos(sumtheta))
    v = np.dot(L, np.sin(sumtheta))
    
    y = np.sqrt(u**2 + v**2)
    return(y)