import numpy as np

def chen_sandu(xx):
    ##########################################################################
    #URRIN ET AL. (2010) 1-DIMENSIONAL FUNCTION, SCALED
    #Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    #Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    ########################################################################
    
    x1 = xx[0]
    x2 = xx[1]
    
    fact1 = np.cos(np.array([x1, x2]))
    fact2 = np.exp(x1*x2)
    
    y = fact1 * fact2
    
    return y