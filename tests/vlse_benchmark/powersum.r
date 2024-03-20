powersum <- function(xx, b)
{
  ##########################################################################
  #
  # POWER SUM FUNCTION
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
  # b  = d-dimensional vector (optional), with default value
  #      c(8, 18, 44, 114) (when d=4)
  #
  ##########################################################################
  
  d <- length(xx)
  ii <- c(1:d)
  
  if (missing(b)) {
    if (d == 4){
      b <- c(8, 18, 44, 114)
    }
    else {
      stop('Value of the d-dimensional vector b is required.')
    }
  }
  
  xxmat <- matrix(rep(xx,times=d), d, d, byrow=TRUE)
  inner <- rowSums(xxmat^ii)
  outer <- sum((inner-b)^2)
	
  y <- outer
  return(y)
}
