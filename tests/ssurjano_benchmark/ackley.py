import numpy as np
def ackley(xx, a=20, b=0.2, c=2*np.pi):
    ##########################################################################
    #ACKLEY FUNCTION
    ############################
    
    d = len(xx)
    sum1 = np.sum(xx**2)
    sum2 = np.sum(np.cos(c*xx))
    
    term1 = -a * np.exp(-b*np.sqrt(sum1/d))
    term2 = -np.exp(sum2/d)
    
    y = term1 + term2 + a + np.exp(1)
    
    return(y)