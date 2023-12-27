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

from ast import Num

from matplotlib.pylab import norm
from .utility import *
import numpy as np
import time
import math
from .rbf import RbfModel
from .optimize import Minimize_Merit_Function

from multiprocessing import Pool


def wrapper_func(xxx_todo_changeme):
    (x, objfunc) = xxx_todo_changeme
    time_start = time.time()
    ret_value = objfunc(x)
    ret_time = time.time() - time_start
    return ret_value, ret_time


def LocalStochRBFstop(data, maxeval, NumberNewSamples):
    """LocalStochRBFstop is the local optimization routine. It iterates at most
    until totally maxeval points have been evaluated, or, if a local minimum
    has been found, the routine terminates in less than maxeval evaluations,
    but will restart from scratch to use up all remaining function evaluation
    points.

    Input:
    Data: struct-variable with problem information (variable bounds,
           objective/simulation function handle, etc.)
    maxeval: maximum number of function evaluations for each trial
    NumberNewSamples: number of points where objective/simulation function
                       is evaluated in every iteration of the algorithm; if
                       NumberNewSamples > 1, evaluations are in parallel, in
                       which case Matlab Parallel Computing Toolbox is
                       required.

    Output:
    Data: updated struct-variable containing the results of the current run
           until stop at local minimum, or stop after maxeval evaluations
    """

    xrange = data.xup - data.xlow  # variable range in every dimension
    minxrange = np.amin(xrange)  # smallest variable range
    m = data.S.shape[0]  # number of points already sampled
    # maxeval = 5
    # scale design points to actual dimensions
    data.S = np.multiply(np.tile(data.xup - data.xlow, (m, 1)), data.S) + np.tile(
        data.xlow, (m, 1)
    )

    data.m = min(
        m, maxeval
    )  # in case number of point in initial experimental design exceed max. number of allowed evaluations
    data.fevaltime = np.zeros(
        maxeval
    )  # initialize vector with time for function evaluations
    data.Y = np.zeros(maxeval)  # initialize array with function values
    data.S = data.S[: data.m, :]  # in case Data.m>maxeval, throw additional points away
    if maxeval > data.m:
        # initialize array with sample points (first Data.m points are from initial experimental design)
        data.S = np.concatenate(
            (data.S, np.zeros((maxeval - data.m, data.dim))), axis=0
        )

    # For serial evaluation of points in initial starting desing:
    # --------------------------SERIAL------------------------------------------
    for ii in range(data.m):  # go through all Data.m points
        time1 = time.time()  # start timer for recording function evaluation time
        res = data.objfunction(np.array(data.S[ii, :]))  # expensive simulation
        data.fevaltime[ii] = time.time() - time1  # record time for expensive evaluation
        data.Y[ii] = res
        if ii == 0:  # initialize best point found so far = first evaluated point
            data.xbest = data.S[0, :]
            data.Fbest = data.Y[ii]
        else:
            if data.Y[ii] < data.Fbest:
                data.Fbest = data.Y[ii]
                data.xbest = data.S[ii, :]
    # --------------------------END SERIAL----------------------------------------
    # for parallel evaluation of points in initial starting design delete
    # comments in the following and comment out the serial code above
    # --------------------------PARALLEL------------------------------------------
    # m = data.m
    # Y = data.Y
    # S = data.S
    # Time = data.fevaltime

    # pool = Pool()
    # pool_res = pool.map_async(wrapper_func, ((i, data.objfunction) for i in S.tolist()))
    # result = pool_res.get()
    # for ii in range(len(result)):
    # Y[ii, 0] = result[ii][0]
    # Time[ii, 0] = result[ii][1]
    # data.Fbest = np.amin(Y[0:m])
    # IDfbest = np.argmin(Y[0:m])
    # data.xbest = S(IDfbest, :)

    # data.Y = Y
    # data.fevaltime = Time
    # --------------------------END PARALLEL--------------------------------------

    # initial RBF matrices
    rbf_model = RbfModel()
    rbf_model.type = data.phifunction
    rbf_model.polynomial = data.polynomial
    rbf_model.sampled_points = data.S[0 : data.m, :]
    PHI = np.zeros((maxeval, maxeval))
    PHI[0 : data.m, 0 : data.m] = rbf_model.eval_phi_sample()
    P = rbf_model.get_ptail(data.S)
    phi0 = rbf_model.phi(0.0)  # Phi-value of two equal points.
    pdim = P.shape[1]
    # tolerance parameters
    data.tolerance = 0.001 * minxrange * np.linalg.norm(np.ones((1, data.dim)))

    # algorithm parameters
    sigma_stdev_default = 0.2 * minxrange
    sigma_stdev = sigma_stdev_default  # current mutation rate
    maxshrinkparam = 5  # maximal number of shrikage of standard deviation for normal distribution when generating the candidate points
    failtolerance = max(5, data.dim)
    succtolerance = 3

    # initializations
    iterctr = 0  # number of iterations
    shrinkctr = 0  # number of times sigma_stdev was shrunk
    failctr = 0  # number of consecutive unsuccessful iterations
    localminflag = 0  # indicates whether or not xbest is at a local minimum
    succctr = 0  # number of consecutive successful iterations

    p = data.Y[0]
    # do until max number of f-evals reached or local min found
    while data.m < maxeval and localminflag == 0:
        iterctr = iterctr + 1  # increment iteration counter
        print("\n Iteration: %d \n" % iterctr)
        print("\n fEvals: %d \n" % data.m)
        print("\n Best value in this restart: %d \n" % data.Fbest)

        # number of new samples in an iteration
        NumberNewSamples = min(NumberNewSamples, maxeval - data.m)

        # replace large function values by the median of all available function values
        Ftransform = np.copy(data.Y[0 : data.m])
        medianF = np.median(data.Y[0 : data.m])
        Ftransform[Ftransform > medianF] = medianF

        # fit the response surface
        # Compute RBF parameters
        a_part1 = np.concatenate(
            (PHI[0 : data.m, 0 : data.m], P[0 : data.m, :]), axis=1
        )
        a_part2 = np.concatenate(
            (np.transpose(P[0 : data.m, :]), np.zeros((pdim, pdim))), axis=1
        )
        a = np.concatenate((a_part1, a_part2), axis=0)

        eta = math.sqrt((1e-16) * np.linalg.norm(a, 1) * np.linalg.norm(a, np.inf))
        coeff = np.linalg.solve(
            (a + eta * np.eye(data.m + pdim)),
            np.concatenate((Ftransform, np.zeros(pdim))),
        )

        # llambda is not a typo, lambda is a python keyword
        data.llambda = coeff[0 : data.m]
        data.ctail = coeff[data.m : data.m + pdim]
        # -------------------------------------------------------------------------------------
        # select the next function evaluation point:
        # introduce candidate points
        x = np.tile(data.xlow, (data.Ncand, 1))
        y = np.tile(data.xbest, (data.Ncand, 1)) + sigma_stdev * np.random.randn(
            data.Ncand, data.dim
        )
        z = np.tile(data.xup, (data.Ncand, 1))
        CandPoint = np.maximum(x, np.minimum(y, z))

        # Init RBF model
        rbf_model = RbfModel()
        rbf_model.type = data.phifunction
        rbf_model.polynomial = data.polynomial
        rbf_model.sampled_points = data.S[0 : data.m, :]
        CandValue, NormValue = rbf_model.eval(CandPoint, data.llambda, data.ctail)
        selindex, distNewSamples = Minimize_Merit_Function(
            CandPoint,
            CandValue,
            np.min(NormValue, axis=0),
            NumberNewSamples,
            data.tolerance,
        )
        xselected = np.reshape(
            CandPoint[selindex, :], (selindex.size, CandPoint.shape[1])
        )
        normval = np.reshape(
            NormValue[:, selindex], (NormValue.shape[0], selindex.size)
        ).T
        assert NumberNewSamples == xselected.shape[0]

        # more than one new point, do parallel evaluation
        # instead of parfor in MATLAB, multiprocessing pool is used here
        if xselected.shape[0] > 1:
            Fselected = np.zeros(xselected.shape[0])
            Time = np.zeros(xselected.shape[0])

            pool = Pool()
            pool_res = pool.map_async(
                wrapper_func, ((i, data.objfunction) for i in xselected.tolist())
            )
            pool.close()
            pool.join()
            result = pool_res.get()
            for ii in range(len(result)):
                Fselected[ii] = result[ii][0]
                Time[ii] = result[ii][1]

            data.fevaltime[data.m : data.m + xselected.shape[0]] = Time
            data.S[data.m : data.m + xselected.shape[0], :] = xselected
            data.Y[data.m : data.m + xselected.shape[0]] = Fselected
            data.m = data.m + xselected.shape[0]
        else:
            time1 = time.time()
            Fselected = data.objfunction(xselected)
            data.fevaltime[data.m] = time.time() - time1
            data.S[data.m, :] = xselected
            data.Y[data.m] = Fselected
            data.m = data.m + 1

        # determine best one of newly sampled points
        minSelected = np.amin(Fselected)
        IDminSelected = np.argmin(Fselected)
        xMinSelected = xselected[IDminSelected, :]
        if minSelected < data.Fbest:
            if data.Fbest - minSelected > (1e-3) * math.fabs(data.Fbest):
                # "significant" improvement
                failctr = 0
                succctr = succctr + 1
            else:
                failctr = failctr + 1
                succctr = 0
            data.xbest = xMinSelected
            data.Fbest = minSelected
        else:
            failctr = failctr + 1
            succctr = 0

        # check if algorithm is in a local minimum
        shrinkflag = 1
        if failctr >= failtolerance:
            if shrinkctr >= maxshrinkparam:
                shrinkflag = 0
                print(
                    "Stopped reducing sigma because the maximum reduction has been reached."
                )
            failctr = 0

            if shrinkflag == 1:
                shrinkctr = shrinkctr + 1
                sigma_stdev = sigma_stdev / 2
                print("Reducing sigma by a half!")
            else:
                localminflag = 1
                print(
                    "Algorithm is probably in a local minimum! Restarting the algorithm from scratch."
                )

        if succctr >= succtolerance:
            sigma_stdev = min(2 * sigma_stdev, sigma_stdev_default)
            succctr = 0
        # update PHI matrix only if planning to do another iteration
        if data.m < maxeval and localminflag == 0:
            n_old = data.m - xselected.shape[0]
            for kk in range(xselected.shape[0]):
                # print kk
                # print n_old+kk
                new_phi = rbf_model.phi(
                    np.concatenate((normval[kk], distNewSamples[kk, 0:kk]))
                )
                PHI[n_old + kk, 0 : n_old + kk] = new_phi
                PHI[0 : n_old + kk, n_old + kk] = new_phi
                PHI[n_old + kk, n_old + kk] = phi0
                P[n_old + kk, 1 : data.dim + 1] = xselected[kk, :]
    data.S = data.S[0 : data.m, :]
    data.Y = data.Y[0 : data.m]
    data.fevaltime = data.fevaltime[0 : data.m]
    data.NumberFevals = data.m

    return data
