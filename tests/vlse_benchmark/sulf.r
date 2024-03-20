sulf <- function(xx)
{
  ##########################################################################
  #
  # SULFUR MODEL FUNCTION
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
  # OUTPUT AND INPUT:
  #
  # DeltaF = direct radiative forcing by sulfate aerosols
  # xx = c(Tr, Ac, Rs, beta_bar, Psi_e, f_Psi_e, Q, Y, L)
  #
  ##########################################################################
  
  Tr       <- xx[1]
  Ac       <- xx[2]
  Rs       <- xx[3]
  beta_bar <- xx[4]
  Psi_e    <- xx[5]
  f_Psi_e  <- xx[6]
  Q        <- xx[7]
  Y        <- xx[8]
  L        <- xx[9]
  
  S0 <- 1361
  A  <- 5 * 10^14
    
  fact1 <- (S0^2) * (1-Ac) * (Tr^2) * (1-Rs)^2 * beta_bar * Psi_e * f_Psi_e
  fact2 <- 3*Q*Y*L / A
  
  DeltaF <- -1/2 * fact1 * fact2
  return(DeltaF)
}
