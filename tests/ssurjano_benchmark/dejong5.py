import numpy as np
def dejong5(xx):
    x1, x2 = xx
    
    A = np.zeros((2, 25))
    a = [-32, -16, 0, 16, 32]
    A[:, :-1] = np.tile(a, (5, 5))
    
    sumterm1 = np.arange(1, 26)
    sumterm2 = np.power((x1 - A[0, :-1]), 6)
    sumterm3 = np.power((x2 - A[1, :-1]), 6)
    sum = np.sum(1 / (sumterm1 + sumterm2 + sumterm3))
    
    y = 1 / (0.002 + sum)
    return y