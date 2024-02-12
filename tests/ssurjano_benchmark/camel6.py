import numpy as np
def camel6(xx):
    ##########################################################################\n # SIX-HUMP CAMEL FUNCTION\n ####################\n \n x1 = xx[0]
    x2 = xx[1]
    term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
    term2 = x1*x2
    term3 = (-4+4*x2**2) * x2**2
    y = term1 + term2 + term3
    return y