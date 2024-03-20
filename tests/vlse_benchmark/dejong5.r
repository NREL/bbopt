dejong5 <- function(xx)
{
  ##########################################################################
  #
  # DE JONG FUNCTION N. 5
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
  # xx = c(x1, x2)
  #
  ##########################################################################
  
  x1 <- xx[1]
  x2 <- xx[2]
	
  A = matrix(0, 2, 25)
  a <- c(-32, -16, 0, 16, 32)
  A[1,] <- rep(a, times=5)
  A[2,] <- rep(a, each=5)
	
  sumterm1 <- c(1:25)
  sumterm2 <- (x1 - A[1,1:25])^6
  sumterm3 <- (x2 - A[2,1:25])^6
  sum <- sum(1 / (sumterm1+sumterm2+sumterm3))
	
  y <- 1 / (0.002 + sum)
  return(y)
}
