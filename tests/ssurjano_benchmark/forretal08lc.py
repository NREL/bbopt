import numpy as np
def forretal08lc(x, A=0.5, B=10, C=-5):
    source('forretal08.r')
    yh = forretal08(x)
    
    term1 = A * yh
    term2 = B * (x-0.5)
    y = term1 + term2 - C
    return y