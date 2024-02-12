import numpy as np

def curretal88exp(xx):
    ##########################################################################
    # CURRIN ET AL. (1988) EXPONENTIAL FUNCTION
    #Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    #Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    ########################################################################
    
    x1 = xx[0]
    x2 = xx[1]
    
    fact1 = 1 - np.exp(-1/(2*x2))
    fact2 = 2300*x1**3 + 1900*x1**2 + 2092*x1 + 60
    fact3 = 100*x1**3 + 500*x1**2 + 4*x1 + 20
    
    y = fact1 * fact2/fact3
    
    return y