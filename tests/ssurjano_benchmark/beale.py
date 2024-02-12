import numpy as np
def beale(xx):
    ##########################################################################
    #BEALE FUNCTION
    ############################
    
    x1, x2 = xx[:-1], xx[-1]
    term1 = (1.5 - x1 + x1*x2)**2
    term2 = (2.25 - x1 + x1*x2**2)**2
    term3 = (2.625 - x1 + x1*x2**3)**2
    
    y = term1 + term2 + term3
    
    return(y)