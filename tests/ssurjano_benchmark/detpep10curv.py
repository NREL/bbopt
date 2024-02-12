import numpy as np
def detpep10curv(xx):
    ##########################################################################\n # DETTE & PEPELYSHEV (2010) CURVED FUNCTION\n ############################\n \n x1 = xx[1]
x2 = xx[2]
x3 = xx[3]
term1 = 4 * ((x1 - 2 + 8*x2 - 8*x2**2)**2)
term2 = (3 - 4*x2)**2
term3 = 16 * np.sqrt(x3+1) * (2*x3-1)**2
y = term1 + term2 + term3
return y