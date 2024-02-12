import numpy as np

def linketal06nosig(xx):
    ##########################################################################
    # LINKLETTER ET AL. (2006) NO SIGNAL FUNCTION
    # Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca
    ########################################################################
    
    # INPUT: xx = array([xx1, xx2, ..., xx10])
    #        where xx1, xx2, ..., xx10 are real numbers
    # OUTPUT: y = 0.0
    ########################################################################
    
    m = 10
    b = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
    C = np.array([4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0,
                  4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6,
                  4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0,
                  4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6])
    C = np.reshape(C, (-1))
    
    xxmat = np.tile(xx, (m, 4)).T
    inner = np.sum((np.square(xxmat - C[:, :-1]).flatten(), axis=0)
    
    outer = 1 / (inner + b)
    
    y = -outer
    
    return y