import numpy as np
def borehole(xx):
    xx_array = np.array(xx)
    
    rw, r, Tu, Hu, Tl, Hl, L, Kw = xx_array[:, :8]
    
    frac1 = 2 * np.pi * Tu * (Hu-Hl)
    
    frac2a = 2*L*Tu / (np.log(r/rw)*rw**2*Kw)
    frac2b = Tu / Tl
    frac2 = np.log(r/rw) * (1+frac2a+frac2b)
    
    y = frac1 / frac2
    return y