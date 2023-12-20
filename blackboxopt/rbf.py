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
from enum import Enum
import scipy.spatial as scp

RbfType = Enum("RbfType", ["LINEAR", "CUBIC", "THINPLATE"])
RbfPolynomial = Enum("RbfPolynomial", ["NONE", "CONSTANT", "LINEAR", "QUADRATIC"])


class RbfModel:
    type: RbfType = RbfType.CUBIC
    polynomial: RbfPolynomial = RbfPolynomial.QUADRATIC

    def phi(self, r):
        """Applies the function phi to the distance r.

        Parameters
        ----------
        r : array_like
            Distances between 2 points.

        Returns
        -------
        out: array_like
            Phi-value according to RBF model.
        """
        if self.type == RbfType.LINEAR:
            return r
        elif self.type == RbfType.CUBIC:
            return np.power(r, 3)
        elif self.type == RbfType.THINPLATE:
            return np.where(
                r > 0,
                np.multiply(np.power(r, 2), np.log(r + np.finfo(np.double).tiny)),
                0,
            )
        else:
            raise ValueError("Unknown rbf_type")

    def InitialRBFMatrices(self, maxeval: int, sampled_points):
        """Set up matrices for computing parameters of RBF model based on points in the initial experimental design.

        Parameters
        ----------
        maxeval : int
            Maximum number of allowed function evaluations.
        sampled_points : array_like
            m-by-d matrix with the sampled points. one point per row, where d is the dimension of the sampled space.

        Returns
        -------
        numpy.matrix
            Matrix containing pairwise distances of all points to each other, will be updated in following iterations.
        numpy.matrix
            PHI-value of two equal points (depends on RBF model!).
        numpy.matrix
            Sample site matrix, needed for determining parameters of the polynomial tail.
        int
            Dimension of P-matrix (number of columns).
        """
        m = sampled_points.shape[0]
        if sampled_points.ndim == 1:
            dim = 1
        elif sampled_points.ndim == 2:
            dim = sampled_points.shape[1]
        else:
            raise ValueError(
                "Invalid matrix size for sampled points. It must be either a 1D or 2D array"
            )

        # Determine pairwise distance between points
        PHI = np.zeros((maxeval, maxeval))
        PHI[0:m, 0:m] = scp.distance.cdist(
            sampled_points[0:m, :], sampled_points[0:m, :], "euclidean"
        )
        # Update PairwiseDistance based on the RBF model
        PHI[0:m, 0:m] = self.phi(PHI[0:m, 0:m])

        # phi-value where the distance of 2 points = 0 (diagonal entries)
        phi0 = self.phi(0)

        # Set up the polynomial tail matrix P
        if self.polynomial == RbfPolynomial.NONE:
            pdim = 0
            P = np.array([])
        elif self.polynomial == RbfPolynomial.CONSTANT:
            pdim = 1
            P = np.ones((maxeval, 1))
        elif self.polynomial == RbfPolynomial.LINEAR:
            pdim = dim + 1
            P = np.concatenate((np.ones((maxeval, 1)), sampled_points), axis=1)
        elif self.polynomial == RbfPolynomial.QUADRATIC:
            pdim = ((dim + 1) * (dim + 2)) // 2
            P = np.concatenate(
                (
                    np.concatenate((np.ones((maxeval, 1)), sampled_points), axis=1),
                    np.zeros((maxeval, (dim * (dim + 1)) // 2)),
                ),
                axis=1,
            )
        else:
            raise ValueError("Invalid polynomial tail")

        return np.asmatrix(PHI), phi0, np.asmatrix(P), pdim
