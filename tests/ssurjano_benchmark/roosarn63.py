import numpy as np
def roosarn63(xx):
    ##########################################################################
    #ROOS & ARNOLD (1963) FUNCTION
    ############################
    
    prod = np.abs(4*xx - 2).prod()
    
    return(prod)