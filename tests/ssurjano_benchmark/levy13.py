import numpy as np

def levy13(xx):
    ##########################################################################
    # LEVY FUNCTION N. 13
    # Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    ########################################################################
    
    x1 = xx[0]
    x2 = xx[1]
    
    term1 = np.power(np.sin(3*np.pi*x1), 2)
    term2 = (x1-1)**2 * (1+(np.sin(3*np.pi*x2))**2)
    term3 = (x2-1)**2 * (1+(np.sin(2*np.pi*x2))**2)
    
    y = term1 + term2 + term3
    
    return y