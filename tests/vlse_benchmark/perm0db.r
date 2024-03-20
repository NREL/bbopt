perm0db <- function(xx, b=10)
{
  ##########################################################################
  #
  # PERM FUNCTION 0, d, beta
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
  # INPUTS:
  #
  # xx = c(x1, x2)
  # b = constant (optional), with default value 10
  #
  ##########################################################################
  
  d <- length(xx)
  ii <- c(1:d)
  jj <- matrix(rep(ii,times=d), d, d, byrow=TRUE)

  xxmat <- matrix(rep(xx,times=d), d, d, byrow=TRUE)
  inner <- rowSums((jj+b)*(xxmat^ii-(1/jj)^ii))	
  outer <- sum(inner^2)
	
  y <- outer
  return(y)
}
