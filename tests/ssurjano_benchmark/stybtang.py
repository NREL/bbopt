import numpy as np
def stybtang(xx):
    ###############################################################
    # STYBLINSKI-TANG FUNCTION
    # Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    ################################################################
    
    sum = np.sum(xx**4 - 16*xx**2 + 5*xx)
    
    y = sum/2
    
    return(y)