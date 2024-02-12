import numpy as np
def morretal06(xx, k1=2):
    ##########################################################################
    #MORRIS ET AL. (2006) FUNCTION
    ############################
    
    alpha = np.sqrt(12) - 6*np.sqrt(0.1)*np.sqrt(k1-1)
    beta = 12 * np.sqrt(0.1)/np.sqrt(k1-1)
    
    sum1 = np.sum(xx[:k1])
    term1 = alpha*sum1
    
    sum2 = np.sum(xx[1:k1]**2, axis=1)
    term2 = beta*np.sum(sum2, axis=0)
    
    y = term1 + term2
    
    return(y)