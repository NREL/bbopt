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
import scipy.spatial as scp
from .phi import phi
from .utility import myException


def ComputeRBF(CandPoint, data):
    """ComputeRBF predicts the objective function values of the candidate points
    and also returns the distance of each candidate point to all already
    sampled points

    Input:
    CandPoint: (Ncand x dimension) matrix with candidate points for next
    expensive function evaluation
    Data: struct-variable with all problem information

    Output:
    RBFVALUE: objective function value predicted by RBF model
    NORMVALUE: matrix with distances of all candidate points to already
    sampled points
    """
    numpoints = CandPoint.shape[0]  # determine number of candidate points
    # compute pairwise distances between candidates and already sampled points
    Normvalue = np.transpose(scp.distance.cdist(CandPoint, data.S[0 : data.m, :]))

    # compute radial basis function value for distances
    U_Y = phi(Normvalue, data.phifunction)

    # determine the polynomial tail (depending on rbf model)
    if data.polynomial == "none":
        PolyPart = np.zeros((numpoints, 1))
    elif data.polynomial == "constant":
        PolyPart = data.ctail * np.ones((numpoints, 1))
    elif data.polynomial == "linear":
        PolyPart = (
            np.concatenate((np.ones((numpoints, 1)), CandPoint), axis=1) * data.ctail
        )
    elif data.polynomial == "quadratic":
        PolyPart = np.concatenate(
            (
                np.concatenate((np.ones((numpoints, 1)), CandPoint), axis=1),
                np.zeros((numpoints, (data.dim * (data.dim + 1)) // 2)),
            ),
            axis=1,
        )
    else:
        raise myException("Error: Invalid polynomial tail.")

    RBFvalue = np.asmatrix(U_Y).T * np.asmatrix(data.llambda) + PolyPart

    return RBFvalue, np.asmatrix(Normvalue)
