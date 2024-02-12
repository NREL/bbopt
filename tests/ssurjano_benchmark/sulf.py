import numpy as np

def sulf(xx):
    Tr, Ac, Rs, beta_bar, Psi_e, f_Psi_e, Q, Y, L = xx[:, 0], xx[:, 1], xx[:, 2], xx[:, 3], xx[:, 4], xx[:, 5], xx[:, 6], xx[:, 7], xx[:, 8]
    
    S0 = 1361
    A = 5 * 10**14
    
    fact1 = (S0**2) * (1-Ac) * (Tr**2) * (1-Rs)**2 * beta_bar * Psi_e * f_Psi_e
    fact2 = 3*Q*Y*L / A
    
    DeltaF = -0.5 * fact1 * fact2
    return(DeltaF)