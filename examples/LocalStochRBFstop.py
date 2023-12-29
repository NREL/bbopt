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
from StochasticRBF import *
from blackboxopt.utility import *
from blackboxopt.rbf import RbfPolynomial, RbfType, RbfModel
from blackboxopt.optimize import minimize

if __name__ == "__main__":
    np.random.seed(3)

    try:
        print("This is the test for LocalStochRBFstop")

        data_file = "datainput_hartman3"
        maxeval = 200
        Ntrials = 3
        PlotResult = 1
        NumberNewSamples = 2
        data = read_check_data_file(data_file)
        data.Ncand = 500 * data.dim
        data.phifunction = RbfType.CUBIC
        data.polynomial = RbfPolynomial.LINEAR
        m = 2 * (data.dim + 1)
        numstart = 0  # collect all objective function values of the current trial here
        Y_all = []  # collect all sample points of the current trial here
        S_all = []  # best objective function value found so far in the current trial
        value = (
            np.inf
        )  # best objective function value found so far in the current trial
        numevals = 0  # number of function evaluations done so far
        Fevaltime_all = []  # collect all objective function evaluation times of the current trial here

        rank_P = 0
        while rank_P != data.dim + 1:
            data.S = SLHDstandard(data.dim, m)
            P = np.concatenate((np.ones((m, 1)), data.S), axis=1)
            rank_P = np.linalg.matrix_rank(P)
        data.S = np.array(
            [
                [0.0625, 0.3125, 0.0625],
                [0.1875, 0.5625, 0.8125],
                [0.3125, 0.8125, 0.3125],
                [0.4375, 0.0625, 0.5625],
                [0.5625, 0.9375, 0.4375],
                [0.6875, 0.1875, 0.6875],
                [0.8125, 0.4375, 0.1875],
                [0.9375, 0.6875, 0.9375],
            ]
        )

        print(data.xlow)
        print(data.xup)
        print(data.objfunction)
        print(data.dim)
        print(data.Ncand)
        # print(data.phifunction)
        # print(data.polynomial)
        print(data.S)
        print("LocalStochRBFstop Start")

        data.S = np.add(np.multiply(data.xup - data.xlow, data.S), data.xlow)

        rbfModel = RbfModel()
        rbfModel.type = data.phifunction
        rbfModel.polynomial = data.polynomial
        rbfModel.x = data.S
        rbfModel.m = data.S.shape[0]

        optres = minimize(
            data.objfunction,
            bounds=(
                (data.xlow[0], data.xup[0]),
                (data.xlow[1], data.xup[1]),
                (data.xlow[2], data.xup[2]),
            ),
            maxeval=maxeval - numevals,
            maxit=1,
            surrogateModel=rbfModel,
            nCandidatesPerIteration=data.Ncand,
            newSamplesPerIteration=NumberNewSamples,
        )

        print("Results")
        print("xlow", data.xlow)
        print("xup", data.xup)
        print("S", optres.samples.shape)
        print("m", optres.samples.shape[0])
        print("Y", optres.fsamples.shape)
        print("xbest", optres.x)
        print("Fbest", optres.fx)
        print("lambda", rbfModel.m)
        print("ctail", rbfModel._beta.shape)
        print("NumberFevals", optres.nfev)

    except myException as e:
        print(e.msg)
