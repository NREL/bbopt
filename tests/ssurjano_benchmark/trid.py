import numpy as np

def trid(xx):
    ##########################################################################
    #URRIN ET AL. (2013) TRID FUNCTION\n
    #Authors: Sonja Surjanovic, Simon Fraser University\n
    #          Derek Bingham, Simon Fraser University\n
    #Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.\n
    ####################################################################
    
    xx = np.array(xx)
    
    n = len(xx)
    xi = xx[2:]
    xold = xx[:-1]
    
    sum1 = (xx[0]-1)**2 + np.sum((xi-1)**2)
    sum2 = np.sum(xi*xold)
    y = sum1 - sum2
    
    return(y)