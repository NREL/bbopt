import numpy as np

def permdb(xx, b=0.5):
    ##########################################################################
    #URRIN ET AL. (2013) PERM FUNCTION d, beta\n
    #Authors: Sonja Surjanovic, Simon Fraser University\n
    #          Derek Bingham, Simon Fraser University\n
    #Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.\n
    ####################################################################
    
    xx = np.array(xx)
    b = b
    
    n = len(xx)
    ii = np.arange(1, n+1)
    jj = np.tile(ii, (n, n))
    
    xxmat = np.tile(xx, (n, n))
    inner = np.sum((jj**ii + b)*((xxmat/jj)**ii - 1), axis=0)
    outer = np.sum(inner**2)
    
    y = outer
    return(y)