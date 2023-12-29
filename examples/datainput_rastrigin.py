"""TODO: <one line to give the program's name and a brief idea of what it does.>
"""

# Copyright (C) 2023 National Renewable Energy Laboratory
# Copyright (C) 2013 Cornell University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# ----------------*****  Contact Information *****--------------------------
#   Primary Contact (Implementation Questions, Bug Reports, etc.):
#   Juliane Mueller: juliane.mueller2901@gmail.com
#
#   Secondary Contact:
#       Christine A. Shoemaker: cas12@cornell.edu
#       Haoyu Jia: leonjiahaoyu@gmail.com
from blackboxopt.utility import *
import numpy as np


def datainput_rastrigin():
    data = Data()
    n = 10
    data.xlow = np.zeros(n)
    data.xup = np.ones(n)
    data.objfunction = myfun
    data.dim = n
    return data


def myfun(x):
    n = 10
    xlow = np.asarray(-10 * np.ones(n))
    xup = np.asarray(10 * np.ones(n))
    x = xlow + np.multiply(xup - xlow, x)
    y = 10 * n + sum(pow(x, 2) - 10 * np.cos(2 * np.pi * x))

    return y


if __name__ == "__main__":
    print(myfun(np.array([[0.5, 0.9]])))
