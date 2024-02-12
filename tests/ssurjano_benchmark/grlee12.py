import numpy as np

def grlee12(x):
    ##########################################################################
    #URRIN ET AL. (2013) GRAMACY & LEE FUNCTION\n
    #Authors: Sonja Surjanovic, Simon Fraser University\n
    #          Derek Bingham, Simon Fraser University\n
    #Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.\n
    ####################################################################
    
    term1 = np.sin(np.array([10*np.pi*x])) / (2*x)
    term2 = (x-1)**4
    
    y = term1 + term2
    
    return y