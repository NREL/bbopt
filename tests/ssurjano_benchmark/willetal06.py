import numpy as np

def willietal06(xx):
    ##########################################################################
    # WILLIAMS ET AL. (2006) FUNCTION
    # Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    ###################################================#####################
    
    x1 = xx[0]
    x2 = xx[1:3]*np.pi
    x3 = xx[3]
    
    term1 = (x1+1) * np.cos(x2)
    y = term1 + 0*x3
    return y