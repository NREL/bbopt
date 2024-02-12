def powell(xx):
    ##########################################################################
    # POWELL FUNCTION
    #######################################################################
    
    # Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    # Copyright 2013.DerekBingham,SimonFraserUniversity.
    
    # There is no warranty, express or implied. We do not assume any liability
    # for the use of this software. If software is modified to produce derivative works, such modified software should be clearly marked. Additionally, this program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; version 2.0 of the License. Accordingly, this program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
    
    # For function details and reference information, see:
    # http://www.sfu.ca/~ssurjano/
    
    #######################################################################
    # INPUT:
    # xx = c(x1, x2, ..., xd)
    #######################################################################
    
    d = len(xx)
    
    xxa = xx[::4]
    xxb = xx[1::4]
    xxc = xx[3::4]
    xxd = xx[-4:]
    
    sumterm1 = (xxa + 10*xxb)**2
    sumterm2 = 5 * (xxc - xxd)**2
    sumterm3 = (xxb - 2*xxc)**4
    sumterm4 = 10 * (xxa - xxd)**4
    
    y = np.sum(np.array([sumterm1, sumterm2, sumterm3, sumterm4]))
    
    return(y)