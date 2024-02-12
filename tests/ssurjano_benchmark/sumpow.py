import numpy as np
def sumpow(xx):
    ##########################################################################
    #SUM OF DIFFERENT POWERS FUNCTION
    #####################################################################
    
    ii = np.arange(1, len(xx)+1)
    sum <- np.sum((np.abs(xx))**(ii+1))
    
    y = sum
    
    return(y)