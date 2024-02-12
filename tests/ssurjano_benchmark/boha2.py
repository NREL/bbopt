import numpy as np

def boha2(xx):
    ##########################################################################
    #BOHACHEVSKY FUNCTION 2
    #Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    #Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    ########################################################################
    
    xx = np.array(xx)
    
    x1 = xx[0]
    x2 = xx[1]
    
    term1 = x1**2
    term2 = 2*x2**2
    term3 = -0.3 * np.cos(3*np.pi*x1) * np.cos(4*np.pi*x2)
    
    y = term1 + term2 + term3 + 0.3
    
    return(y)