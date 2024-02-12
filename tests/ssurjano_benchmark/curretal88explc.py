import numpy as np
def curretal88explc(xx):
    x1, x2 = xx[:2]
    
    maxarg = np.maximum([0, x2-1/20])
    
    yh1 = curretal88exp(np.array([x1+1/20, x2+1/20]))
    yh2 = curretal88exp(np.array([x1+1/20, maxarg]))
    yh3 = curretal88exp(np.array([x1-1/20, x2+1/20]))
    yh4 = curretal88exp(np.array([x1-1/20, maxarg]))
    
    y = (yh1 + yh2 + yh3 + yh4) / 4
    return y