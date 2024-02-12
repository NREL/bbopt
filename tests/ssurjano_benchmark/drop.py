import numpy as np
def drop(xx):
    ##########################################################################\n # DROP-WAVE FUNCTION\n ############################\n \n x1 = xx[0]
x2 = xx[1]
frac1 = 1 + np.cos(12*np.sqrt(x1**2+x2**2))
frac2 = (x1**2+x2**2) / 2 + 2
y = -frac1/frac2
return y