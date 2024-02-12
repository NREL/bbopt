import numpy as np
def environ(xx, s=np.array([0.5, 1, 1.5, 2, 2.5]), t=np.arange(from=0.3, to=60, by=0.3)):
    ##########################################################################
    # ENVIRONMENTAL MODEL FUNCTION
    ########################################################################
    
    a = 1
    b =5.1/(4*np.pi**2)
    c =5/np.pi
    r =6
    s =10
    t =1/(8*np.pi)
    
    x1,x2 =xx[:2],xx[2:]
    
    term1 =a * (x2 - b*x1**2 + c*x1 - r)**2
    term2 =s*(1-t)*np.cos(x1)
    
    y =term1 + term2 + s + 5*x1
    return(y)