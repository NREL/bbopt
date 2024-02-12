import numpy as np

def perm0db(xx, b=10):
    ##########################################################################
    #PERM FUNCTION 0, d, beta
    #Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    #Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    ########################################################################
    
    xx = np.array(xx)
    
    n = len(xx)
    ii = np.arange(1, n+1)
    jj = np.tile(ii, (n, 2)) + b*np.ones((n, 2), dtype=int32)
    
    xxmat = np.tile(xx, (n, 1))
    inner = np.sum((jj+b)*(xxmat**ii - 1/jj)**2, axis=0)
    outer = np.sum(inner)
    
    return(outer)