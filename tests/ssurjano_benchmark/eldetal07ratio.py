import numpy as np
def eldetal07ratio(xx):
    ##################################################################
    # ELDRED ET AL. (2007) FUNCTION
    # Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    ################################################################
    
    x1 = xx[0]
    x2 = xx[1]
    
    y = x1 / x2
    
    return(y)