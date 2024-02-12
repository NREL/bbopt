import numpy as np

def shortcol(xx, b=5, h=15):
    ##########################################################################
    #SHORT COLUMN FUNCTION
    #Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    #Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    ##########################################################################\n  # FOR THE USE OF THIS SOFTWARE.  If software is modified to produce# # derivative works, such modified software should be clearly marked.# # Additionally, this program is free software; you can redistribute it # # and/or modify it under the terms of the GNU General Public License as # # published by the Free Software Foundation; version 2.0 of the License. # # Accordingly, this program is distributed in the hope that it will be # # useful, but WITHOUT ANY WARRANTY; without even the implied warranty # # of MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. See the GNU # # General Public License for more details.#
    ##########################################################################\n # For function details and reference information, see:#
    ##########################################################################\n # http://www.sfu.ca/~ssurjano/#
    ##########################################################################\n #
    ##########################################################################\n # INPUTS:#
    ##########################################################################\n # xx = c(Y, M, P)#
    ##########################################################################\n # b  = width of cross-section (optional), with nominal value 5#
    ##########################################################################\n # h  = depth of cross-section (optional), with nominal value 15#
    ##########################################################################\n #
    ##########################################################################\n # OUTPUTS:#
    ##########################################################################\n # y = short column factor#
    ##########################################################################\n #
    ##########################################################################\n #
    Y, M, P = xx[:3]
    b = int(b) if b else 5
    h = int(h) if h else 15
    term1 = -4*M / (b*(h**2)*Y)
    term2 = -P**2 / ((b**2)*(h**2)*(Y**2))
    y = 1 + term1 + term2
    return y