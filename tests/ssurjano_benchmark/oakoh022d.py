import numpy as np

def oakoh022d(xx):
    ##########################################################################
    #URRIN ET AL. (2002) 2-DIMENSIONAL FUNCTION, SCALED
    #Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    #Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    ########################################################################
    
    x1 = xx[0]
    x2 = xx[1]
    
    term1 = x1 + x2
    term2 = 2*np.cos(xx[0]) + 2*np.sin(xx[1])
    
    y = 5 + term1 + term2
    
    return y