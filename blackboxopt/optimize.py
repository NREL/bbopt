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


def Minimize_Merit_Function(
    x: np.ndarray,
    fx: np.ndarray,
    dist: np.ndarray,
    n: int,
    tol: float,
    weightpattern: np.ndarray = np.array([0.3, 0.5, 0.8, 0.95]),
) -> tuple[np.ndarray, np.ndarray]:
    """Select n points based on their values and distances to candidates.

    The points are chosen from x such that they minimize the expression
    :math:`w f_s(x) + (1-w) (-d_s(x))`, where
    - `w` is a weight.
    - `f_s(x)` is the estimated value for the objective function on x, scaled to [0,1].
    - `d_s(x)` is the minimum distance between x and the previously selected evaluation points, scaled to [-1,0].

    If there are more than one new sample point to be
    selected, the distances of the candidate points to the previously
    selected candidate point have to be taken into account.

    Parameters
    ----------
    x : numpy.ndarray
        Matrix with candidate points.
    fx : numpy.ndarray
        Esimated values for the objective function on each candidate point.
    dist : numpy.ndarray
        Minimum distance between a candidate point and previously evaluated sampled points.
    n : int
        Number of points to be selected for the next costly evaluation.
    tol : float
        Tolerance value for excluding candidate points that are too close to already sampled points.
    weightpattern: np.ndarray
        Weight(s) `w` to be used in the score given in a circular list.

    Returns
    -------
    numpy.ndarray
        Vector with indexes of the selected points.
    numpy.ndarray
        n-by-n lower triangular matrix with the distances between the selected points.
    """
    assert fx.ndim == 1

    selindex = np.zeros(n, dtype=np.intp)
    distNewSamples = np.zeros((n, n))

    # Scale function values to [0,1]
    minval = np.amin(fx)
    maxval = np.amax(fx)
    if minval == maxval:
        scaledvalue = np.ones(fx.size)
    else:
        scaledvalue = (fx - minval) / (maxval - minval)

    def argminscore(
        dist: np.ndarray, valueweight: float = weightpattern[-1]
    ) -> np.intp:
        """Gets the index of the candidate point that minimizes the score.

        Parameters
        ----------
        dist : numpy.ndarray
            Minimum distance between a candidate point and previously evaluated sampled points.
        valueweight: float
            Weight `w` to be used in the score.

        Returns
        -------
        numpy.intp
            Index of the selected candidate.
        """
        # Scale distance values to [0,1]
        maxdist = np.amax(dist)
        mindist = np.amin(dist)
        if maxdist == mindist:
            scaleddist = np.ones(dist.size)
        else:
            scaleddist = (maxdist - dist) / (maxdist - mindist)

        # Compute weighted score for all candidates
        score = valueweight * scaledvalue + (1 - valueweight) * scaleddist

        # Assign bad values to points that are too close to already
        # evaluated/chosen points
        score[dist < tol] = np.inf

        # Return index with the best (smallest) score
        return np.argmin(score)

    if n == 1:
        selindex[0] = argminscore(dist)
    else:
        selindex[0] = argminscore(dist, weightpattern[0])
        for ii in range(1, n):
            # compute distance of all candidate points to the previously selected
            # candidate point
            newDist = scp.distance.cdist(
                x[selindex[ii - 1], :].reshape(1, x.shape[1]), x, "euclidean"
            )[0]
            dist = np.minimum(dist, newDist)

            selindex[ii] = argminscore(dist, weightpattern[ii % 4])

            for j in range(ii - 1):
                distNewSamples[ii, j] = np.linalg.norm(
                    x[selindex[ii], :] - x[selindex[j], :]
                )
            distNewSamples[ii, ii - 1] = newDist[selindex[ii]]

    return selindex, distNewSamples
