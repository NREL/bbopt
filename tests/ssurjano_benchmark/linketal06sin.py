import numpy as np

def linketal06sin(xx):
    ##########################################################################
    # LINKLETTER ET AL. (2006) SINUSOIDAL FUNCTION
    # Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    ########################################################################
    
    x1 = xx[0]
    x2 = xx[1:] * 5
    
    y = np.sin(x1) + np.sin(x2)
    return y