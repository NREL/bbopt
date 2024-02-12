import numpy as np


def park91alc(xx):
    ##########################################################################
    #
    # PARK (1991) FUNCTION 1, LOWER FIDELITY CODE
    # Calls: park91a.r
    # This function, from Xiong et al. (2013), is used as the "low-accuracy
    # code" version of the function park91a.r.
    #
    # Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    #
    # Copyright 2013. Derek Bingham, Simon Fraser University.
    #
    # THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
    # FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
    # derivative works, such modified software should be clearly marked.
    # Additionally, this program is free software; you can redistribute it
    # and/or modify it under the terms of the GNU General Public License as
    # published by the Free Software Foundation; version 2.0 of the License.
    # Accordingly, this program is distributed in the hope that it will be
    # useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    # of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    # General Public License for more details.
    #
    # For function details and reference information, see:
    # http://www.sfu.ca/~ssurjano/
    #
    ##########################################################################
    #
    # INPUT:
    #
    # xx = c(x1, x2, x3, x4)
    #
    ##########################################################################

    # Assume park91a.r contains the function park91a
    # Import or define park91a function

    # Assuming park91a is a function that returns yh based on xx
    def park91a(xx):
        # Example placeholder function, replace with actual implementation
        return np.sum(xx)  # Example placeholder implementation

    x1 = xx[0]
    x2 = xx[1]
    x3 = xx[2]
    x4 = xx[3]

    yh = park91a(xx)

    term1 = (1 + np.sin(x1) / 10) * yh
    term2 = -2 * x1 + x2**2 + x3**2

    y = term1 + term2 + 0.5
    return y


# Test the function
xx_test = np.random.rand(4)  # Example input
result = park91alc(xx_test)
print("Result:", result)
