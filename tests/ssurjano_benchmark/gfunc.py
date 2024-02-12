import numpy as np
def gfunc(xx, a=np.array([0]*len(xx))):
    new1 = np.abs(4*xx-2) + a
    new2 = 1 + a
    prod = np.prod(new1/new2)
    
    y = prod
    return(y)