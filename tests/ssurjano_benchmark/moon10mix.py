import numpy as np

def moon10mix(xx):
    ##########################################################################
    # MOON (2010) MIXED FUNCTION\n  #\n # Authors: Sonja Surjanovic, Simon Fraser University\n #          Derek Bingham, Simon Fraser University\n # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.\n #\n # Copyright 2013. Derek Bingham, Simon Fraser University.\n #\n # THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY\n # FOR THE USE OF THIS SOFTWARE.  If software is modified to produce\n # derivative works, such modified software should be clearly marked.\n # Additionally, this program is free software; you can redistribute it \n # and/or modify it under the terms of the GNU General Public License as \n # published by the Free Software Foundation; version 2.0 of the License. \n # Accordingly, this program is distributed in the hope that it will be \n # useful, but WITHOUT ANY WARRANTY; without even the implied warranty \n # of MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. See the GNU \n # General Public License for more details.\n #\n # For function details and reference information, see:\n # http://www.sfu.ca/~ssurjano/\n #\n #######################################################################
    x1 = xx[0]
    x2 = xx[1]
    z  = xx[2]
    
    term1 = (x1+x2) * np.where(z==1, True, False)
    term2 = (3*x1) * np.where(z==2, True, False)
    
    y = term1 + term2
    return(y)