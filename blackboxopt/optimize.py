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


def Minimize_Merit_Function(
    CandPoint: np.ndarray,
    CandValue: np.ndarray,
    NormValue: np.ndarray,
    NumberNewSamples: int,
    tol: float,
) -> np.ndarray | np.intp:
    """Select points for next costly evaluation and Computes the distance and response surface
    criteria for every candidate point. The values are scaled to [0,1], and
    the candidate with the best weighted score of both criteria becomes the
    new sample point. If there are more than one new sample point to be
    selected, the distances of the candidate points to the previously
    selected candidate point have to be taken into account.

    Parameters
    ----------
    CandPoint : numpy.ndarray
        Matrix with candidate points.
    CandValue : numpy.ndarray
        Esimated values for the objective function on each candidate point.
    NormValue : numpy.ndarray
        Distances between candidate points and previously evaluated sampled points.
    NumberNewSamples : int
        Number of points to be selected for the next costly evaluation.
    tol : float
        Tolerance value for excluding candidate points that are too close to already sampled points.

    Returns
    -------
    numpy.ndarray
        Matrix with all selected points for the next evaluation.
    dict
        Distances to previously evaluated points and other selected candidate points.
    """
    assert CandValue.ndim == 1

    MinCandValue = np.amin(CandValue)
    MaxCandValue = np.amax(CandValue)

    if MinCandValue == MaxCandValue:
        ScaledCandValue = np.ones(CandValue.size)
    else:
        ScaledCandValue = (CandValue - MinCandValue) / (MaxCandValue - MinCandValue)

    if NumberNewSamples == 1:
        valueweight = 0.95
        CandMinDist = np.amin(NormValue, axis=0)
        MaxCandMinDist = np.amax(CandMinDist)
        MinCandMinDist = np.amin(CandMinDist)
        if MaxCandMinDist == MinCandMinDist:
            ScaledCandMinDist = np.ones(CandMinDist.size)
        else:
            ScaledCandMinDist = (MaxCandMinDist - CandMinDist) / (
                MaxCandMinDist - MinCandMinDist
            )

        # compute weighted score for all candidates
        CandTotalValue = (
            valueweight * ScaledCandValue + (1 - valueweight) * ScaledCandMinDist
        )

        # assign bad scores to candidate points that are too close to already sampled
        # points
        CandTotalValue[CandMinDist < tol] = np.inf

        return np.argmin(CandTotalValue)
    else:  # more than one new sample point wanted
        selindex = np.zeros(NumberNewSamples, dtype=int)

        wp_id = -1
        weightpattern = np.array([0.3, 0.5, 0.8, 0.95])
        for ii in range(NumberNewSamples):
            wp_id = wp_id + 1
            if wp_id > 3:
                wp_id = 0
            valueweight = weightpattern[wp_id]
            # This part is the same as NumberNewSamples == 1 case
            # This coding style is poor, but this is how they wrote the MATLAB code..
            if ii == 0:  # select first candidate point
                CandMinDist = np.amin(NormValue, axis=0)
                MaxCandMinDist = np.amax(CandMinDist)
                MinCandMinDist = np.amin(CandMinDist)
                if MaxCandMinDist == MinCandMinDist:
                    ScaledCandMinDist = np.ones(CandMinDist.size)
                else:
                    ScaledCandMinDist = (MaxCandMinDist - CandMinDist) / (
                        MaxCandMinDist - MinCandMinDist
                    )

                # compute weighted score for all candidates
                CandTotalValue = (
                    valueweight * ScaledCandValue
                    + (1 - valueweight) * ScaledCandMinDist
                )

                # assign bad scores to candidate points that are too close to already sampled
                # points
                CandTotalValue[CandMinDist < tol] = np.inf

                selindex[0] = np.argmin(CandTotalValue)
            else:
                # compute distance of all candidate points to the previously selected
                # candidate point
                NormValueP = np.sqrt(
                    np.sum(
                        (
                            np.tile(
                                CandPoint[selindex[ii - 1], :],
                                (CandPoint.shape[0], 1),
                            )
                            - CandPoint
                        )
                        ** 2,
                        axis=1,
                    )
                )

                # re-scale distance values to [0,1]
                CandMinDist = np.minimum(CandMinDist, NormValueP)
                MaxCandMinDist = np.amax(CandMinDist)
                MinCandMinDist = np.amin(CandMinDist)
                if MaxCandMinDist == MinCandMinDist:
                    ScaledCandMinDist = np.ones(CandMinDist.size)
                else:
                    ScaledCandMinDist = (MaxCandMinDist - CandMinDist) / (
                        MaxCandMinDist - MinCandMinDist
                    )

                # compute weighted score for all candidates
                CandTotalValue = (
                    valueweight * ScaledCandValue
                    + (1 - valueweight) * ScaledCandMinDist
                )

                # assign bad values to points that are too close to already
                # evaluated/chosen points
                CandTotalValue[CandMinDist < tol] = np.inf

                selindex[ii] = np.argmin(CandTotalValue)

        return selindex
