def wingweight(xx):
    ##########################################################################
    # WING WEIGHT FUNCTION
    #######################################################################
    
    authors = "Sonja Surjanovic, Simon Fraser University\nDerek Bingham, Simon Fraser University"
    copyright = "Copyright 2013. Derek Bingham, Simon Fraser University.\nTHERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY FOR THE USE OF THIS SOFTWARE. If software is modified to produce derivative works, such modified software should be clearly marked.\nAdditionally, this program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; version 2.0 of the License.\nAccordingly, this program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details."
    reference = "http://www.sfu.ca/~ssurjano/"
    
    #######################################################################
    # OUTPUT AND INPUT:
    #######################################################################
    
    y = wingweight(xx)
    xx = [xx[1], xx[2], xx[3], xx[4] * np.pi/180, xx[5], xx[6], xx[7], xx[8], xx[9], xx[10]]
    
    Sw = xx[0]
    Wfw = xx[1]
    A = xx[2]
    LamCaps = xx[3]
    q = xx[4]
    lam = xx[5]
    tc = xx[6]
    Nz = xx[7]
    Wdg = xx[8]
    Wp = xx[9]
    
    fact1 = 0.036 * Sw**0.758 * Wfw**0.0035
    fact2 = ((A / np.cos(Lamaps))**2)**0.6
    fact3 = q**0.006 * lam**0.04
    fact4 = (100*tc / np.cos(Lamaps))**(-0.3)
    fact5 = (Nz*Wdg)**0.49
    
    term1 = Sw * Wp
    
    y = fact1*fact2*fact3*fact4*fact5 + term1
    
    return(y)