##########################################################################
#
# MARTHE DATASET READ-IN
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
# OBSERVATIONS: 300
#
# INPUT VARIABLES:
#
# per1  = hydraulic conductivity layer 1
# per2  = hydraulic conductivity layer 2
# per3  = hydraulic conductivity layer 3
# perz1 = hydraulic conductivity zone 1
# perz2 = hydraulic conductivity zone 2
# perz3 = hydraulic conductivity zone 3
# perz4 = hydraulic conductivity zone 4
# d1    = longitudinal dispersivity layer 1
# d2    = longitudinal dispersivity layer 2
# d3    = longitudinal dispersivity layer 3
# dt1   = transversal dispersivity layer 1
# dt2   = transversal dispersivity layer 2
# dt3   = transversal dispersivity layer 3
# kd1   = volumetric distribution coefficient 1.1
# kd2   = volumetric distribution coefficient 1.2
# kd3   = volumetric distribution coefficient 1.3
# poros = porosity
# i1    = infiltration type 1
# i2    = infiltration type 2
# i3    = infiltration type 3
#
# OUTPUT VARIABLES:
#
# p102k
# p104
# p106
# p2.76
# p29k
# p31k
# p35k
# p37k
# p38
# p4b
#
##########################################################################

marthedata <- read.table('marthedata.txt', as.is=TRUE, fill=TRUE, header=TRUE)

per1  <- marthedata$per1
per2  <- marthedata$per2
per3  <- marthedata$per3
perz1 <- marthedata$perz1
perz2 <- marthedata$perz2
perz3 <- marthedata$perz3
perz4 <- marthedata$perz4
d1    <- marthedata$d1
d2    <- marthedata$d2
d3    <- marthedata$d3
dt1   <- marthedata$dt1
dt2   <- marthedata$dt2
dt3   <- marthedata$dt3
kd1   <- marthedata$kd1
kd2   <- marthedata$kd2
kd3   <- marthedata$kd3
poros <- marthedata$poros
i1    <- marthedata$i1
i2    <- marthedata$i2
i3    <- marthedata$i3

p102K <- marthedata$p102K
p104  <- marthedata$p104
p106  <- marthedata$p106
p2.76 <- marthedata$p2.76
p29K  <- marthedata$p29K
p31K  <- marthedata$p31K
p35K  <- marthedata$p35K
p37K  <- marthedata$p37K
p38   <- marthedata$p38
p4b   <- marthedata$p4b
