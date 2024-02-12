import numpy as np

def camel3(xx):
    ##########################################################################
    #URRIN ET AL. (2013) THREE-HUMP CAMEL FUNCTION, SCALED
    #Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    #Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    ########################################################################
    
    x1 = xx[0]
    x2 = xx[1]
    
    term1 = 2*x1**2
    term2 = -1.05*x1**4
    term3 = x1**6 / 6
    term4 = x1*x2
    term5 = x2**2
    
    y = term1 + term2 + term3 + term4 + term5
    
    return y