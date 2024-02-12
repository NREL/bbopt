import numpy as np

def schaffer2(xx):
    x1 = xx[1]
    x2 = xx[2]
    
    fact1 = (np.sin(x1**2-x2**2))**2 - 0.5
    fact2 = (1 + 0.001*(x1**2+x2**2))**2
    
    y = 0.5 + fact1/fact2
    return(y)