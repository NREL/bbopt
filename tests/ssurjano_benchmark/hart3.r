hart3 <- function(xx) {
    ##########################################################################
    # 
    # HARTMANN 3-DIMENSIONAL FUNCTION
    #
    # Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    #
    # Copyright 2013. Derek Bingham, Simon Fraser University.
    #
    # THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
    # FOR THE USE OF THIS SOFTWARE. If software is modified to produce
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
    # xx = c(x1, x2, x3)
    #
    ##########################################################################

    alpha <- c(1.0, 1.2, 3.0, 3.2)
    A <- c(3.0, 10, 30,
           0.1, 10, 35,
           3.0, 10, 30,
           0.1, 10, 35)
    A <- matrix(A, 4, 3, byrow=TRUE)
    P <- 10^(-4) * c(3689, 1170, 2673,
                     4699, 4387, 7470,
                     1091, 8732, 5547,
                     381, 5743, 8828)
    P <- matrix(P, 4, 3, byrow=TRUE)

    xxmat <- matrix(rep(xx, times=4), 4, 3, byrow=TRUE)
    inner <- rowSums(A[,1:3] * (xxmat - P[,1:3])^2)
    outer <- sum(alpha * exp(-inner))

    y <- -outer
    return(y)
}
