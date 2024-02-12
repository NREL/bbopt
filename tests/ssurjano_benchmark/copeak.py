import numpy as np
def copeak(xx, u=np.array([0.5]*len(xx)), a=np.array([5]*len(xx))):
    d = len(xx)
    sum = np.sum(a*xx)
    
    y = (1 + sum)**(-d-1)
    return(y)