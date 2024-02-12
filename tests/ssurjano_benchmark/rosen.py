import numpy as np
def rosen(xx):
    d = len(xx)
    xi = xx[:d-1]
    xnext = xx[d-1:]
    
    sum = 100*(xnext - xi**2)**2 + (xi - 1)**2
    
    y = sum
    return y