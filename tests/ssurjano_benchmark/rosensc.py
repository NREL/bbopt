import numpy as np
def rosensc(xx):
    xx_array = np.array(xx)
    
    xxbar = (15*xx - 5).reshape(-1, 3)
    xnextbar = xxbar[1:4]
    
    sum = np.sum((100*(xnextbar-xibar**2)**2 + (1 - xibar)**2))
    
    y = ((sum - 3.827*1e5) / (3.755*1e5))
    return y