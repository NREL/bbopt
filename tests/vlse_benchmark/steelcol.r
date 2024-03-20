steelcol <- function(xx)
{
  ##########################################################################
  #
  # STEEL COLUMN FUNCTION
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
  # xx = c(Fs, P1, P2, P3, B, D, H, F0, E)
  #
  ##########################################################################
  
  Fs <- xx[1]
  P1 <- xx[2]
  P2 <- xx[3]
  P3 <- xx[4]
  B  <- xx[5]
  D  <- xx[6]
  H  <- xx[7]
  F0 <- xx[8]
  E  <- xx[9]
  
  L <- 7500
  
  P   <- P1 + P2 + P3
  Eb <- (pi^2)*E*B*D*(H^2) / (2*(L^2))
  
  term1 <- 1 / (2*B*D)
  term2 <- F0*Eb / (B*D*H*(Eb-P))
  
  y <- Fs - P*(term1 + term2)
  return(y)
}
