# ----------------********************************--------------------------
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
# ----------------********************************--------------------------


# ----------------*****  Contact Information *****--------------------------
#   Primary Contact (Implementation Questions, Bug Reports, etc.):
#   Juliane Mueller: juliane.mueller2901@gmail.com
#
#   Secondary Contact:
#       Christine A. Shoemaker: cas12@cornell.edu
#       Haoyu Jia: leonjiahaoyu@gmail.com
# ----------------********************************--------------------------
import numpy as np
import math
from utility import myException


def phi(r, type):
    """determines phi-value of distance r between 2 points (depends on chosen RBF model)

    Input:
         r: distance between 2 points
         type: RBF model type

    Output:
         output: phi-value according to RBF model
    """
    if type == "linear":
        output = r
    elif type == "cubic":
        output = np.power(r, 3)
    elif type == "thinplate":
        if r >= 0:
            output = np.multiply(np.power(r, 2), math.log(r + np.finfo(np.double).tiny))
        else:
            output = np.zeros(r.shape)
    else:
        raise myException("Error: Unkonwn type.")

    return output
