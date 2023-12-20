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
import math
from .utility import myException, phi


def InitialRBFMatrices(maxeval, data, PairwiseDistance):
    """set up matrices for computing parameters of RBF model based on points in initial experimental design

    Input:
    maxevals: maximal number of allowed function evaluations
    Data: struct-variable with all problem information such as sampled points
    PairwiseDistance: pairwise distances between points in initial experimental design

    Output:
    PHI: matrix containing pairwise distances of all points to each other, will be updated in following iterations
    phi0: PHI-value of two equal points (depends on RBF model!)
    P: sample site matrix, needed for determining parameters of polynomial tail
    pdim: dimension of P-matrix (number of columns)
    """
    PHI = np.zeros((maxeval, maxeval))
    if data.phifunction == "linear":
        PairwiseDistance = PairwiseDistance
    elif data.phifunction == "cubic":
        PairwiseDistance = PairwiseDistance**3
    elif data.phifunction == "thinplate":
        PairwiseDistance = PairwiseDistance**2 * math.log(
            PairwiseDistance + np.finfo(np.double).tiny
        )

    PHI[0 : data.m, 0 : data.m] = PairwiseDistance
    phi0 = phi(
        0, data.phifunction
    )  # phi-value where distance of 2 points =0 (diagonal entries)

    if data.polynomial == "None":
        pdim = 0
        P = np.array([])
    elif data.polynomial == "constant":
        pdim = 1
        P = np.ones((maxeval, 1)), data.S
    elif data.polynomial == "linear":
        pdim = data.dim + 1
        P = np.concatenate((np.ones((maxeval, 1)), data.S), axis=1)
    elif data.polynomial == "quadratic":
        pdim = (data.dim + 1) * (data.dim + 2) // 2
        P = np.concatenate(
            (
                np.concatenate((np.ones((maxeval, 1)), data.S), axis=1),
                np.zeros((maxeval, (data.dim * (data.dim + 1)) // 2)),
            ),
            axis=1,
        )
    else:
        raise myException("Error: Invalid polynomial tail.")
    return np.asmatrix(PHI), np.asmatrix(phi0), np.asmatrix(P), pdim
