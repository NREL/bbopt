import numpy as np
def morcaf95b(xx):
    ##########################################################################
    #MOROKOFF & CAFLISCH (1995) FUNCTION 2
    ############################
    
    d = len(xx)
    fact1 = 1 / ((d-0.5)**d)
    prod = np.prod(d-xx)
    y = fact1 * prod
    
    return(y)