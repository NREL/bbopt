import numpy as np
def park91a(xx):
    ##########################################################################
    #PARK (1991) FUNCTION 1
    ############################
    
    x1, x2, x3, x4 = xx[:-1], xx[-1]
    term1a = x1 / 2
    term1b = np.sqrt(1 + (x2+x3**2)*x4/(x1**2)) - 1
    term1 = term1a * term1b
    
    term2a = x1 + 3*x4
    term2b = np.exp(1 + np.sin(x3))
    term2 = term2a * term2b
    
    y = term1 + term2
    
    return(y)