import numpy as np
def prpeak(xx, u=np.array([0.5]), a=np.array([5])):
    ##########################################################################
    #PRODUC PEAK INTEGRAND FAMILY
    ############################
    
    prod = 1 / (1/(a**2) + ((xx-u)**2))
    y = prod
    
    return(y)