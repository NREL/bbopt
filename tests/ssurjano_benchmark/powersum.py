import numpy as np
def powersum(xx, b=None):
    d = len(xx)
    
    if b is None:
        if d == 4:
            b = np.array([8, 18, 44, 114])
        else:
            raise ValueError("Value of the d-dimensional vector b is required.")
            
    xx_array = np.array(xx)
    ii = np.arange(1, d+1)
    
    if isinstance(b, int):
        b_array = np.array([b]*d)
    else:
        b_array = b
        
    xxmat = np.tile(xx_array, (d, 1))
    inner = np.sum(np.power(xxmat, ii), axis=0)
    outer = np.sum((inner - b_array)**2)
    
    y = outer
    return y