import numpy as np
def bukin6(xx):
    ##########################################################################
    #BUKIN FUNCTION N. 6
    ############################
    
    x1, x2 = xx[:,0], xx[:,1]
    term1 = 100 * np.sqrt(np.abs(x2 - 0.01*x1**2))
    term2 = 0.01 * np.abs(x1+10)
    
    y = term1 + term2
    
    return(y)