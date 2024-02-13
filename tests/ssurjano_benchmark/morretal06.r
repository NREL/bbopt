morretal06 <- function(xx, k1=2)
{
  ##########################################################################
  #
  # MORRIS ET AL. (2006) FUNCTION
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
  # xx = c(x1, x2, ..., x30)
  # k1 = number of arguments with an effect (optional), with default value
  #      2  
  #
  ##########################################################################

  alpha <- sqrt(12) - 6*sqrt(0.1)*sqrt(k1-1)
  beta <- 12 * sqrt(0.1) / sqrt(k1-1)

  sum1 <- 0
  for (ii in 1:k1)
  {
    sum1 <- sum1 + xx[ii]
  }
  term1 <- alpha * sum1

  sum2 <- 0
  for (ii in 1:(k1-1))
  {
    for (jj in (ii+1):k1)
    {
      sum2 <- sum2 + xx[ii]*xx[jj]
    }
  }
  term2 <- beta * sum2

  y <- term1 + term2
  return(y)
}
