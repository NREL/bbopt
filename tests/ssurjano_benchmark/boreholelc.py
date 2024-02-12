import numpy as np

def boreholelc(xx):
    rw = xx[0]
    r = xx[1]
    Tu = xx[2]
    Hu = xx[3]
    Tl = xx[4]
    Hl = xx[5]
    L = xx[6]
    Kw = xx[7]
    
    frac1 = 5 * Tu * (Hu-Hl)
    
    frac2a = 2*L*Tu / (np.log(r/rw)*rw**2*Kw)
    frac2b = Tu / Tl
    frac2 = np.log(r/rw) * (1.5+frac2a+frac2b)
    
    y = frac1 / frac2
    return(y)