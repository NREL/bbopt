"""TODO: <one line to give the program's name and a brief idea of what it does.>
Copyright (C) 2023 National Renewable Energy Laboratory
Copyright (C) 2013 Cornell University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


# ----------------*****  Contact Information *****--------------------------
#   Primary Contact (Implementation Questions, Bug Reports, etc.):
#   Juliane Mueller: juliane.mueller2901@gmail.com
#
#   Secondary Contact:
#       Christine A. Shoemaker: cas12@cornell.edu
#       Haoyu Jia: leonjiahaoyu@gmail.com
from blackboxopt.utility import *
from numpy import *
from numpy.matlib import *
from operator import mul


def datainput_Branin():
    data = Data()
    data.xlow = zeros(2)
    data.xup = ones(2)
    data.objfunction = myfun
    data.dim = 2
    return data


def myfun(x):
    xlow = array([-5, 0])
    xup = array([10, 15])
    x = xlow + (xup - xlow) * x
    print(x[0, 0])
    print(x[0, 1])
    y = (
        pow(
            x[0, 1] - 5.1 * pow(x[0, 0], 2) / (4 * pow(pi, 2)) + 5 * x[0, 0] / pi - 6, 2
        )
        + 10 * (1 - 1 / (8 * pi)) * cos(x[0, 0])
        + 10
    )
    # c = array([1, 1.2, 3, 3.2])
    # A = array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    # P = array([[0.3689, 0.1170, 0.2673],
    #    [0.4699, 0.4387, 0.747],
    #    [0.1091, 0.8732, 0.5547],
    #    [0.0382, 0.5743, 0.8828]])
    # y = -sum(c * exp(-sum(A * (repmat(x, 4, 1) - P) ** 2, axis = 1)))
    # print y
    return y


if __name__ == "__main__":
    print(myfun(array([[0.5, 0.9]])))
