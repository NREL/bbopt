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
    sampled_points: np.ndarray = np.array([])

    def get_dim(self) -> int:
        """Get the dimension of the domain space"""
        if self.sampled_points.ndim == 1:
            return 1
        elif self.sampled_points.ndim == 2:
            return self.sampled_points.shape[1]
        else:
            raise ValueError(
                "Invalid matrix size for sampled points. It must be either a 1D or 2D array"
            )

    def phi(self, r):
        """Applies the function phi to the distance r.

        Parameters
        ----------
        r : array_like
            Distance(s) between points.

        Returns
        -------
        out: array_like
            Phi-value of the distances provided on input.
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

    def eval_phi_sample(self, metric="euclidean") -> np.ndarray:
        """Returns a matrix containing the phi-value of the distances of all sampled points to each other.

        Parameters
        ----------
        metric : str or callable, optional
            The distance metric to use. If a string, the distance function can be
            'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
            'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
            'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
            'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
            'sokalsneath', 'sqeuclidean', 'yule'.

        Returns
        -------
        out: numpy.ndarray
            Matrix containing the phi-value of the distances of all sampled points to each other.
        """
        return self.phi(
            scp.distance.cdist(self.sampled_points, self.sampled_points, metric)
        )

    def get_ptail(self, x) -> np.ndarray:
        """Returns a sample site matrix, needed for determining parameters of the polynomial tail.

        Parameters
        ----------
        x : array_like
            Input vector of coordinates

        Returns
        -------
        out: numpy.ndarray
            Site matrix, needed for determining parameters of the polynomial tail.
        """
        m = x.shape[0]
        dim = self.get_dim()

        # Set up the polynomial tail matrix P
        if self.polynomial == RbfPolynomial.NONE:
            return np.array([])
        elif self.polynomial == RbfPolynomial.CONSTANT:
            return np.ones((m, 1))
        elif self.polynomial == RbfPolynomial.LINEAR:
            return np.concatenate((np.ones((m, 1)), x), axis=1)
        elif self.polynomial == RbfPolynomial.QUADRATIC:
            return np.concatenate(
                (
                    np.concatenate((np.ones((m, 1)), x), axis=1),
                    np.zeros((m, (dim * (dim + 1)) // 2)),
                ),
                axis=1,
            )
        else:
            raise ValueError("Invalid polynomial tail")

    def eval(self, x, alpha, beta):
        """Evaluates the model at one or multiple points.

        Parameters
        ----------
        x : array_like
            m-by-d matrix with m point coordinates in a d-dimensional space.
        alpha : array_like
            Coefficients of the RBF model associated to the phi function.
        beta : array_like
            Coefficients of the RBF model associated to the polynomial tail.

        Returns
        -------
        numpy.ndarray
            Value for the RBF model on each of the input points.
        numpy.ndarray
            Matrix with distances of all points to sampled points i the RBF model.
        """
        # compute pairwise distances between candidates and sampled points
        Dist = scp.distance.cdist(self.sampled_points, x)

        if self.polynomial == RbfPolynomial.NONE:
            y = np.matmul(self.phi(Dist).T, alpha)
        else:
            y = np.matmul(self.phi(Dist).T, alpha) + self.get_ptail(x) * beta

        return y, Dist
