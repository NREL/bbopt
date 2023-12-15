#----------------********************************--------------------------
# Copyright (C) 2013 Cornell University
# This file is part of the program StochasticRBF.py
#
#    StochasticRBF.py is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    StochasticRBF.py is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with StochasticRBF.py.  If not, see <http://www.gnu.org/licenses/>.
#----------------********************************--------------------------



#----------------*****  Contact Information *****--------------------------
#   Primary Contact (Implementation Questions, Bug Reports, etc.):
#   Juliane Mueller: juliane.mueller2901@gmail.com
#       
#   Secondary Contact:
#       Christine A. Shoemaker: cas12@cornell.edu
#       Haoyu Jia: leonjiahaoyu@gmail.com
#----------------********************************--------------------------
from utility import *
from numpy import *
from numpy.matlib import *
from operator import mul

def datainput_rastrigin():
    data = Data()
    n=10
    data.xlow = zeros(n)
    data.xup = ones(n)
    data.objfunction = myfun
    data.dim = n
    return data

def myfun(x):
    n=10
    xlow=np.asarray(-10*ones(n))
    xup=np.asarray(10*ones(n))
    x = xlow+(xup-xlow)*x    
    y= 10 * n+ sum(pow(x,2)-10*cos(2*pi*x))
    
    return y

if __name__ == '__main__':
    print myfun(array([[0.5, 0.9]]))
