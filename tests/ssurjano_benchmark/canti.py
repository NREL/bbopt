import numpy as np

def canti(xx, w=4, t=2):
    ##########################################################################
    #URRIN ET AL. (2013) CANTILEVER BEAM FUNCTION\n
    #Authors: Sonja Surjanovic, Simon Fraser University\n
    #          Derek Bingham, Simon Fraser University\n
    #Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.\n
    ####################################################################
    
    R = xx[1]
    E = xx[2]
    X = xx[3]
    Y = xx[4]
    
    L = 100
    D_0 = 2.2535
    
    Sterm1 = 600*Y / (w*(t**2))
    Sterm2 = 600*X / ((w**2)*t)
    
    S = Sterm1 + Sterm2
    
    Dfact1 = 4*(L**3) / (E*w*t)
    Dfact2 = np.sqrt((Y/(t**2))**2 + (X/(w**2))**2)
    
    D = Dfact1 * Dfact2
    
    y = [D, S]
    return(y)