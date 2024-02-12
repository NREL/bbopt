import numpy as np
def schwef(xx):
    ##########################################################################
    #SUM OF DIFFERENT POWERS FUNCTION
    ############################################################
    
    d = len(xx)
    sum <- np.sum((np.abs(xx))**(d+1))*np.sin(np.sqrt(np.abs(xx)))
    
    y = 418.9829*d - sum
    
    return(y)