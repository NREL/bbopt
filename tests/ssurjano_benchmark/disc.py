import numpy as np
def disc(xx, u=np.array([0.5]*len(xx)), a=np.array([5]*len(xx))):
    ##########################################################################\n # DROP-WAVE FUNCTION\n ############################\n \n x1 = xx[1]
x2 = xx[2]
u1 = u[1]
u2 = u[2]
if (x1 > u1 | x2 > u2):
    y = 0
else:
    sum = np.sum(a*xx)
    y = np.exp(sum)
return y