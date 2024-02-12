import numpy as np
def michal(xx, m=10):
    ##########################################################################
    #MICHALEWICZ FUNCTION
    ############################
    
    ii = np.arange(len(xx))
    sum = np.sum(np.sin(xx) * (np.sin(ii*xx**2/np.pi))**(2*m))
    y = -sum
    
    return(y)