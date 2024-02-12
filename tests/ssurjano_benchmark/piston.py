import numpy as np
def piston(xx):
    ##########################################################################\n # PISTON FUNCTION\n ####################\n \n M = xx[1]\n S = xx[2]\n V0 = xx[3]\n k = xx[4]\n P0 = xx[5]\n Ta = xx[6]\n T0 = xx[7]
    Aterm1 = np.array([P0, S])
    Aterm2 = 19.62 * M
    Aterm3 = -k*V0 / S
    A = np.sum(Aterm1 + Aterm2 + Aterm3)
    
    Vfact1 = S / (2*k)
    Vfact2 = np.sqrt(np.array([A, 4*k*(P0*V0/T0)*Ta])**2 + 4*k*(P0*V0/T0)*Ta)
    V = Vfact1 * (Vfact2 - A)
    
    fact1 = M
    fact2 = k + (S**2)*(P0*V0/T0)*(Ta/(V**2))
    
    C = 2 * np.pi * np.sqrt(fact1/fact2)
    return C