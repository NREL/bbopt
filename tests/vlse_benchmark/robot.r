robot <- function(xx)
{
  ##########################################################################
  #
  # ROBOT ARM FUNCTION
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
  # OUTPUT AND INPUTS:
  #
  # y = distance from the end of the arm to the origin
  # xx = c(theta1, theta2, theta3, theta4, L1, L2, L3, L4)
  #
  #########################################################################
  
  theta <- xx[1:4]
  L     <- xx[5:8]
  
  thetamat <- matrix(rep(theta,times=4), 4, 4, byrow=TRUE)
  thetamatlow <- thetamat
  thetamatlow[upper.tri(thetamatlow)] <- 0
  sumtheta <- rowSums(thetamatlow)
  
  u <- sum(L*cos(sumtheta))
  v <- sum(L*sin(sumtheta))
  
  y <- (u^2 + v^2)^(0.5)
  return(y)
}
