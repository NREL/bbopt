import numpy as np
def langer(xx, m=5, cvec=[1,2,5,2,3], A=np.array([[3, 5], [5, 2]])):
    ##########################################################################
    #LANGERMANN FUNCTION
    ############################
    
    d = len(xx)
    
    if np.isnan(cvec):
        raise ValueError("Value of the m-dimensional vector cvec is required.")
        
    if np.isnan(A):
        raise ValueError("Value of the (mxd)-dimensional matrix A is required.")
        
    xxmat = np.tile(xx, (m, d))
    
    inner = np.sum((np.abs(xxmat - A) ** 2), axis=1)
    outer = np.exp(-inner / np.pi) * np.cos(np.pi * inner)
    
    y = np.sum(outer * cvec, axis=0)
    
    return(y)