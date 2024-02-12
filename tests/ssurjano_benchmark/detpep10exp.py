import numpy as np
def detpep10exp(xx):
    ##########################################################################\
    # DETTE & PEPELYSHEV (2010) EXPONENTIAL FUNCTION
    #####################################################################
    
    x1, x2, x3 = xx[:-1], xx[1:]
    
    term1 = np.exp(-2/(x1**1.75))
    term2 = np.exp(-2/(x2**1.5))
    term3 = np.exp(-2/(x3**1.25))
    
    y = 100 * (term1 + term2 + term3)
    
    return(y)