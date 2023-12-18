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
from ComputeRBF import ComputeRBF


def Minimize_Merit_Function(data, CandPoint, NumberNewSamples):
    """Minimize_Merit_Function computes the distance and response surface
    criteria for every candidate point. The values are scaled to [0,1], and
    the candidate with the best weighted score of both criteria becomes the
    new sample point. If there are more than one new sample point to be
    selected, the distances of the candidate points to the previously
    selected candidate point have to be taken into account.

    Input:
    Data: struct-variable with all problem information
    CandPoint: matrix with candidate points
    NumberNewSamples: number of points to be selected for next costly evaluation

    Output:
    xselected: matrix with all seleced points for next evaluation
    normval: cell array with distances to previously evaluated points and
    other selected candidate points, needed for updating PHI matrix later
    """

    CandValue, NormValue = ComputeRBF(CandPoint, data)
    MinCandValue = np.amin(CandValue)
    MaxCandValue = np.amax(CandValue)

    if MinCandValue == MaxCandValue:
        ScaledCandValue = np.ones((CandValue.shape[0], 1))
    else:
        ScaledCandValue = (CandValue - MinCandValue) / (MaxCandValue - MinCandValue)

    normval = {}
    if NumberNewSamples == 1:
        valueweight = 0.95
        CandMinDist = np.asmatrix(np.amin(NormValue, axis=0)).T
        MaxCandMinDist = np.amax(CandMinDist)
        MinCandMinDist = np.amin(CandMinDist)
        if MaxCandMinDist == MinCandMinDist:
            ScaledCandMinDist = np.ones((CandMinDist.shape[0], 1))
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
        CandTotalValue[CandMinDist < data.tolerance] = np.inf

        MinCandTotalValue = np.amin(CandTotalValue)
        selindex = np.argmin(CandTotalValue)
        xselected = np.array(CandPoint[selindex, :])

        # MATLAB code used cell struct here. Here we use Python buildin map to
        # achieve the same functionality, and the key start from 0
        normval[0] = np.asmatrix((NormValue[:, selindex])).T
    else:  # more than one new sample point wanted
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
                CandMinDist = np.asmatrix(np.amin(NormValue, axis=0)).T
                MaxCandMinDist = np.amax(CandMinDist)
                MinCandMinDist = np.amin(CandMinDist)
                if MaxCandMinDist == MinCandMinDist:
                    ScaledCandMinDist = np.ones((CandMinDist.shape[0], 1))
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
                CandTotalValue[CandMinDist < data.tolerance] = np.inf

                MinCandTotalValue = np.amin(CandTotalValue)
                selindex = np.argmin(CandTotalValue)
                xselected = np.array(CandPoint[selindex, :])

                # MATLAB code used cell struct here. Here we use Python buildin map to
                # achieve the same functionality
                # Do not need to transpose here, because slice just give 1d array
                normval[0] = np.asmatrix(NormValue[:, selindex]).T
            else:
                # compute distance of all candidate points to the previously selected
                # candidate point
                NormValueP = np.sqrt(
                    np.sum(
                        np.asarray(
                            (
                                np.tile(xselected[ii - 1, :], (CandPoint.shape[0], 1))
                                - CandPoint
                            )
                        )
                        ** 2,
                        axis=1,
                    )
                )
                NormValue = np.concatenate((NormValue, np.asmatrix(NormValueP)), axis=0)

                # re-scale distance values to [0,1]
                CandMinDist = np.asmatrix(np.amin(NormValue, axis=0)).T
                MaxCandMinDist = np.amax(CandMinDist)
                MinCandMinDist = np.amin(CandMinDist)
                if MaxCandMinDist == MinCandMinDist:
                    ScaledCandMinDist = np.ones((CandMinDist.shape[0], 1))
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
                CandTotalValue[CandMinDist < data.tolerance] = np.inf

                MinCandTotalValue = np.amin(CandTotalValue)
                selindex = np.argmin(CandTotalValue)
                xselected = np.concatenate(
                    (xselected, np.array(CandPoint[selindex, :])), axis=0
                )
                normval[ii] = np.asmatrix(NormValue[:, selindex]).T

    return xselected, normval
