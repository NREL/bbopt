powell <- function(xx)
{
  ##########################################################################
  #
  # POWELL FUNCTION
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
  # xx = c(x1, x2, ..., xd)
  #
  ##########################################################################
  
  d <- length(xx)
	
  xxa <- xx[seq(1, d-3, 4)]
  xxb <- xx[seq(2, d-2, 4)]
  xxc <- xx[seq(3, d-1, 4)]
  xxd <- xx[seq(4, d, 4)]

  sumterm1 <- (xxa + 10*xxb)^2
  sumterm2 <- 5 * (xxc - xxd)^2
  sumterm3 <- (xxb - 2*xxc)^4
  sumterm4 <- 10 * (xxa - xxd)^4
  sum <- sum(sumterm1 + sumterm2 + sumterm3 + sumterm4)
	
  y <- sum
  return(y)
}
