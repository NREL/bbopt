moonetal12 <- function(xx)
{  
  ##########################################################################
  #
  # MOON ET AL. (2012) FUNCTION
  # Calls: borehole.r, wingweight.r, otlcircuit.r, piston.r
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
  # xx = c(x1, x2, ..., x31)
  #
  ##########################################################################
  
  source('borehole.r')
  source('wingweight.r')
  source('otlcircuit.r')
  source('piston.r')
  
  y <- numeric(0)
  y[1] <- borehole(xx[1:8])
  y[2] <- wingweight(xx[9:18])
  y[3] <- otlcircuit(xx[19:24])
  y[4] <- piston(xx[25:31])
  
  miny <- min(y)
  maxy <- max(y)
  
  ystar <- (y-miny) / (maxy-miny)
  
  y <- sum(ystar)
  return(y)
}
