detpep10curv <- function(xx)
{
  ##########################################################################
  #
  # DETTE & PEPELYSHEV (2010) CURVED FUNCTION
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
  # xx = c(x1, x2, x3)
  #
  #########################################################################
  
  x1 <- xx[1]
  x2 <- xx[2]
  x3 <- xx[3]
  
  term1 <- 4 * (x1 - 2 + 8*x2 - 8*x2^2)^2
  term2 <- (3 - 4*x2)^2
  term3 <- 16 * sqrt(x3+1) * (2*x3-1)^2
  
  y <- term1 + term2 + term3
  return(y)
}
