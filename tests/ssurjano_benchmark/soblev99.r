soblev99 <- function(xx, b, c0=0)
{
  ##########################################################################
  #
  # SOBOL' & LEVITAN (1999) FUNCTION
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
  # xx = c(x1, x2, ..., xd)
  # b  = d-dimensional vector (optional), with default value
  #      c(2, 1.95, 1.9, 1.85, 1.8, 1.75, 1.7, 1.65, 0.4228, 0.3077, 0.2169,
  #        0.1471, 0.0951, 0.0577, 0.0323, 0.0161, 0.0068, 0.0021, 0.0004, 0)
  #      (when d<=20)
  # c0 = constant term (optional), with default value 0  
  #
  ##########################################################################
  
  d <- length(xx)
  
  if (missing(b)) {
    if (d <= 20){
      b <- c(2, 1.95, 1.9, 1.85, 1.8, 1.75, 1.7, 1.65, 0.4228, 0.3077, 0.2169, 0.1471, 0.0951, 0.0577, 0.0323, 0.0161, 0.0068, 0.0021, 0.0004, 0)
    }
    else {
      stop('Value of the d-dimensional vector b is required.')
    }
  }
  
  Id <- 1
  for (ii in 1:d) {
    bi  <- b[ii]
    new <- (exp(bi)-1) / bi
    Id  <- Id * new
  }
  
  sum <- 0
  for (ii in 1:d) {
    bi  <- b[ii]
    xi  <- xx[ii]
    sum <- sum + bi*xi
  }
  
  y <- exp(sum) - Id + c0
  return(y)
}
