moon10hdc2 <- function(xx)
{
  #########################################################################
  #
  # MOON (2010) HIGH-DIMENSIONALITY FUNCTION, C-2
  # This function is a modification of the function moon10hd.r.
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
  # xx = c(x1, x2, ..., x20)
  #
  ##########################################################################
  
  coefflin <- c(-2.08, 2.11, 0.76, -0.57, -0.72, -0.47, 0.39, 1.40, -0.09, -0.70, -1.27, -1.03, 1.07, 2.23, 2.46, -1.31, -2.94, 2.63, 0.07, 2.44)
  
  sumdeg1 <- sum(coefflin*xx)
  
  coeffs <- matrix(0, 20, 20)
  coeffs[,1]  <- c(1.42,  2.18, 0.58, -1.21, -7.15, -1.29, -0.19, -2.75, -1.16, -1.09,  0.89, -0.16,  4.43,  1.65, -1.25, -1.35,  1.15, -39.42,  47.44,  1.42)
  coeffs[,2]  <- c(   0, -1.70, 0.84,  1.20, -2.35, -0.16, -0.19, -5.93, -1.15,  1.89, -3.47, -0.07, -0.60, -1.09, -3.23,  0.44,  1.24,   2.13,  -0.71,  1.64)
  coeffs[,3]  <- c(   0,     0, 1.00, -0.49,  1.74,  1.29, -0.35, -4.73,  3.27,  1.87,  1.42, -0.96, -0.91,  2.06,  2.89,  0.25,  1.97,   3.04,   2.00,  1.64)
  coeffs[,4]  <- c(   0,     0,    0, -3.23,  2.75, -1.40,  0.24, -0.70, -0.17, -3.38, -1.87, -0.17,  1.56,  2.40, -1.70,  0.32,  2.11,  -0.20,   1.39, -2.01)
  coeffs[,5]  <- c(   0,     0,    0,     0, -1.10,  2.34, -3.90, -0.80,  0.13, -3.97,  1.99,  0.45,  1.77, -0.50,  1.86,  0.02, -2.08,  -1.78,   1.76,  1.30)
  coeffs[,6]  <- c(   0,     0,    0,     0,     0,  0.21, -0.03, -0.37, -1.27,  2.78,  1.37, -2.75, -3.15,  1.86,  0.12, -0.74,  1.06,  -3.76,  -0.43,  1.25)
  coeffs[,7]  <- c(   0,     0,    0,     0,     0,     0, -4.16,  0.26, -0.30, -2.69, -2.56, 57.98, -2.13,  1.36,  1.45,  3.09, -1.73,  -1.66,  -3.94, -2.56)
  coeffs[,8]  <- c(   0,     0,    0,     0,     0,     0,     0, -1.00,  0.77,  1.09, -1.15, -1.09, -2.74,  1.59,  1.41,  0.48,  2.16,   0.34,   4.17,  0.73)
  coeffs[,9]  <- c(   0,     0,    0,     0,     0,     0,     0,     0,  3.06,  2.46,  5.80, -5.15, -2.05,  3.17,  3.40, -0.49, -6.71,  -0.74,   2.78, -0.41)
  coeffs[,10] <- c(   0,     0,    0,     0,     0,     0,     0,     0,     0,  3.34,  2.36, -1.77, -3.16,  1.89,  2.20, -0.71, -3.78,   0.98,   1.40, -0.59)
  coeffs[,11] <- c(   0,     0,    0,     0,     0,     0,     0,     0,     0,     0, -1.17, -2.45,  6.04,  3.22,  0.19, -0.03, -2.65,  -1.02,  -1.96, -2.66)
  coeffs[,12] <- c(   0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,  1.52,  1.36, -0.59, -1.05, -0.84, -1.30,   0.42,   1.86, -0.32)
  coeffs[,13] <- c(   0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,  0.42, -0.50,  0.21, -0.18,  3.04,  -0.53,  -0.12,  0.09)
  coeffs[,14] <- c(   0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0, -1.13, -2.42, -3.93, -2.30,   0.40,   0.81, -1.10)
  coeffs[,15] <- c(   0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0, -0.26,  5.31,  1.66,  -3.10,   3.37,  4.32)
  coeffs[,16] <- c(   0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0, -2.26,  0.00,  -0.77,  -3.90, -1.08)
  coeffs[,17] <- c(   0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,  0.62,  -1.06,  -0.86,  0.44)
  coeffs[,18] <- c(   0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,   0.35,  -1.99,  1.50)
  coeffs[,19] <- c(   0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,      0, -26.68,  1.34)
  coeffs[,20] <- c(   0,     0,    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,      0,      0, -0.38)
  
  xxmat <- matrix(rep(xx,times=20), 20, 20, byrow=TRUE)
  factors <- rowSums(coeffs*xxmat*t(xxmat))
  sumdeg2 <- sum(sum(factors))
  
  y <- sumdeg1 + sumdeg2
  return(y)
}
