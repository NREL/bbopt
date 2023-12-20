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

__authors__ = ["Juliane Mueller", "Christine A. Shoemaker", "Haoyu Jia"]
__contact__ = "juliane.mueller@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = [
    "Juliane Mueller",
    "Christine A. Shoemaker",
    "Haoyu Jia",
    "Weslley S. Pereira",
]
__version__ = "0.1.0"
__deprecated__ = False

import numpy as np


def SLHDstandard(d: int, m: int) -> np.ndarray:
    """Creates a Symmetric Latin Hypercube Design.

    Parameters
    ----------
    d : int
        Dimension of the input.
    m : int
        Number of initial points to be selected.

    Returns
    -------
    out: numpy.ndarray
        Symmetric Latin Hypercube Design.
    """
    delta = 1.0 / m
    X = np.zeros((m, d))

    # Create the initial design
    for j in range(d):
        for i in range(m):
            X[i, j] = ((2.0 * (i + 1) - 1) / 2.0) * delta

    # Generate permutation matrix P
    P = np.zeros((m, d), dtype=int)
    P[:, 0] = np.arange(m)

    if m % 2 == 0:
        k = m // 2
    else:
        k = (m - 1) // 2
        P[k, :] = (k + 1) * np.ones((1, d))

    for j in range(1, d):
        P[0:k, j] = np.random.permutation(np.arange(k))

        for i in range(k):
            # Use numpy functions for better performance
            if np.random.random() < 0.5:
                P[m - 1 - i, j] = m - 1 - P[i, j]
            else:
                P[m - 1 - i, j] = P[i, j]
                P[i, j] = m - 1 - P[i, j]
    InitialPoints = np.zeros([m, d])
    for j in range(d):
        for i in range(m):
            InitialPoints[i, j] = X[P[i, j], j]
    return InitialPoints


class myException(Exception):
    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg


class Data:
    def __init__(self):
        ## User defined parameters
        self.xlow = np.matrix([])
        self.xup = np.matrix([])
        self.objfunction = None
        self.dim = -1

    def validate(self):
        if self.dim == -1:
            raise myException("You must provide the problem dimension.\n")
        if not isinstance(self.dim, int) or self.dim <= 0:
            raise myException("Dimension must be positive integer.")
        if (
            not isinstance(self.xlow, np.matrix)
            or not isinstance(self.xup, np.matrix)
            or self.xlow.shape != (1, self.dim)
            or self.xup.shape != (1, self.dim)
        ):
            raise myException(
                "Vector length of lower and upper bounds must equal problem dimension\n"
            )
        # print any(self.xlow[0][i] > 0 for i in range(self.dim))
        # print any(self.xlow[0][i] > self.xup[0][i] for i in range(self.dim))
        comp_list = np.less_equal(self.xlow, self.xup).tolist()[0]
        if any(i == False for i in comp_list):
            raise myException("Lower bounds have to be lower than upper bounds.\n")


class Solution:
    def __init__(self):
        self.BestValues = None
        self.BestPoints = None
        self.NumFuncEval = None
        self.AvgFuncEvalTime = None
        self.FuncVal = None
        self.DMatrix = None
        self.NumberOfRestarts = None
