import numpy as np
def zhouetal11(xx):
    x, z = xx[:2]
    
    sign = 1
    
    if z == 1:
        c = 6.8
    elif z == 2:
        c = 7
        sign = -1
    elif z == 3:
        c = 7.2
    
    y = sign * np.cos(np.pi*c*x/2)
    return y