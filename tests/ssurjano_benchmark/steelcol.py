import numpy as np

def steelcol(xx):
    ##########################################################################
    # STEEL COLUMN FUNCTION\n #\n # Authors: Sonja Surjanovic, Simon Fraser University\n #          Derek Bingham, Simon Fraser University\n # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.\n #\n # Copyright 2013. Derek Bingham, Simon Fraser University.\n #\n # THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY\n # FOR THE USE OF THIS SOFTWARE. If software is modified to produce\n # derivative works, such modified software should be clearly marked.\n # Additionally, this program is free software; you can redistribute it \n # and/or modify it under the terms of the GNU General Public License as \n # published by the Free Software Foundation; version 2.0 of the License. \n # Accordingly, this program is distributed in the hope that it will be \n # useful, but WITHOUT ANY WARRANTY; without even the implied warranty \n # of MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. See the GNU \n # General Public License for more details.\n #\n # For function details and reference information, see:\n # http://www.sfu.ca/~ssurjano/\n #\n #######################################################################
    Fs = xx[1]
    P1 = xx[2]
    P2 = xx[3]
    P3 = xx[4]
    B  = xx[5]
    D  = xx[6]
    H  = xx[7]
    F0 <- xx[8]
    E  = xx[9]
    
    L = 7500
    
    P   = P1 + P2 + P3
    Eb = (np.pi**2)*E*B*D*(H**2) / (2*(L**2))
    term1 = np.sqrt(P*F0)/(2*L)
    term2 = -term1
    
    y <- Fs - P*(term1 + term2)
    return(y)