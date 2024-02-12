import numpy as np
def griewank(xx):
    xx_array = np.array(xx)
    
    ii = np.arange(1, len(xx)+1)
    sum = np.sum(xx**2 / 4000)
    prod = np.prod(np.cos(xx/np.sqrt(ii)))
    
    y = sum - prod + 1
    return y