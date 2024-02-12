import numpy as np
def webetal96(xx):
    ##########################################################################
    #WEBSTER ET AL. (1996) FUNCTION
    ############################
    
    A, B = xx[:-1], xx[-1]
    y = np.power(A, 2) + np.power(B, 3)
    
    return(y)