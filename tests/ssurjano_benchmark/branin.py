import numpy as np
def branin(xx, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    ##########################################################################\n # BRANIN FUNCTION\n ####################\n \n x1 = xx[1]
    x2 = xx[2]
    term1 = a * ((x2 - b*xx**2 + c*xx - r)**2)
    term2 = s*(1-t)*np.cos(x1)
    y = term1 + term2 + s
    return y