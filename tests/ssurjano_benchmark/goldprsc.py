import numpy as np

def goldprsc(xx):
    ##########################################################################
    #URRIN ET AL. (1988) GOLDSTEIN-PRICE FUNCTION, SCALED
    #Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    #Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    ########################################################################
    
    x1 = 4*xx[0] - 2
    x2 = 4*xx[1] - 2
    
    fact1a = (x1 + x2 + 1)**2
    fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
    fact1 = 1 + fact1a*fact1b
    
    fact2a = (2*x1 - 3*x2)**2
    fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
    fact2 = 30 + fact2a*fact2b
    
    prod = fact1 * fact2
    
    y = (np.log(prod) - 8.693) / 2.427
    
    return y