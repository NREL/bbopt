hanetal09 <- function(xx, b=c(1, 0, -1, 6, 4, 5, -6, -6, -6))
{
  ##########################################################################
  #
  # HAN ET AL. (2009) FUNCTION
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
  # xx = c(x, z)
  # b = c(b01, b02, b03, b11, b12, b13, b21, b22, b23) (optional)  
  #
  ##########################################################################
  
  x <- xx[1]
  z <- xx[2]
  
  b01 <- b[1]
  b11 <- b[4]
  b21 <- b[7]
  b02 <- b[2]
  b12 <- b[5]
  b22 <- b[8]
  b03 <- b[3]
  b13 <- b[6]
  b23 <- b[9]
  
  if (z == 1)
  {
    y <- b01 + b11*x + b21*x^2
  }
  else if (z == 2)
  {
    y <- b02 + b12*x + b22*x^2
  }
  else if (z == 3)
  {
    y <- b03 + b13*x + b23*x^2
  }
  else
  {
    y <- NA
  }
  
  return(y)
}
