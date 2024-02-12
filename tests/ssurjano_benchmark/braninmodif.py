import numpy as np

def braninmodif(xx):
    ##########################################################################
    # BRANIN FUNCTION, MODIFIED\n #\n # Authors: Sonja Surjanovic, Simon Fraser University\n #          Derek Bingham, Simon Fraser University\n # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.\n #\n # Copyright 2013. Derek Bingham, Simon Fraser University.\n #\n # THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY\n # FOR THE USE OF THIS SOFTWARE. If software is modified to produce\n # derivative works, such modified software should be clearly marked.\n # Additionally, this program is free software; you can redistribute it \n # and/or modify it under the terms of the GNU General Public License as \n # published by the Free Software Foundation; version 2.0 of the License. \n # Accordingly, this program is distributed in the hope that it will be \n # useful, but WITHOUT ANY WARRANTY; without even the implied warranty \n # of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU \n # General Public License for more details.\n #\n # For function details and reference information, see:\n # http://www.sfu.ca/~ssurjano/\n #\n ##########################################################################
    xx = np.array(xx)
    
    a = 1
    b = 5.1/(4*np.pi**2)
    c = 5/np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)
    
    x1 = xx[0]
    x2 = xx[1]
    
    term1 = a * (x2 - b*x1**2 + c*x1 - r)**2
    term2 = s*(1-t)*np.cos(x1)
    
    y = term1 + term2 + s + 5*x1
    return(y)