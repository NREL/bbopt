import numpy as np

def linketal06simple(xx):
    ##########################################################################
    # LINKLETTER ET AL. (2006) SIMPLE FUNCTION
    # Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    # Copyright 2013DerekBinghamSimonFraserUniversity.
    #
    ########################################################################
    
    # INPUT:
    xx = np.array(xx)
    
    x1 = xx[0]
    x2 = xx[1]
    x3 = xx[2]
    x4 = xx[3]
    
    term1 = 0.2*x1 + 0.2*x2
    term2 = 0.2*x3 + 0.2*x4
    
    y = term1 + term2
    
    return(y)