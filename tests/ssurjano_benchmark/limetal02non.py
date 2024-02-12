import numpy as np
def limetal02non(xx):
    x1, x2 = xx[:2]
    
    fact1 = 30 + 5*x1*np.sin(5*x1)
    fact2 = 4 + np.exp(-5*x2)
    
    y = (fact1*fact2 - 100) / 6
    return y