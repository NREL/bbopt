import numpy as np

def park91blc(xx):
    ##########################################################################
    # PARK (1991) FUNCTION 2, LOWER FIDELITY CODE
    #Calls: park91b.py
    #This function, from Xiong et al. (2013), is used as the "low-accuracy" version of the function park91b.r.
    ########################################################################
    
    xx = np.array(xx)
    
    source('park91blc.py')
    yh = park91b(xx)
    
    y = 1.2*yh -1
    
    return(y)