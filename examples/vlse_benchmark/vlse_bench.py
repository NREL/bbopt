"""Run the optimization on the VLSE benchmark."""

# Copyright (c) 2024 Alliance for Sustainable Energy, LLC

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__authors__ = ["Weslley S. Pereira"]
__contact__ = "weslley.dasilvapereira@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = ["Weslley S. Pereira"]
__version__ = "0.4.2"
__deprecated__ = False

import os
import numpy as np
import pickle
import time
from tests.test_vlse_bench import run_optimizer
from blackboxopt import optimize, acquisition, rbf, gp
from pathlib import Path
from sklearn.gaussian_process.kernels import RBF as GPkernelRBF

# Functions to be tested
myRfuncs = (
    "branin",
    "hart3",
    "hart6",
    "shekel",
    "ackley",
    "levy",
    "powell",
    "michal",
    "spheref",
    "rastr",
    "mccorm",
    "bukin6",
    "camel6",
    "crossit",
    "drop",
    "egg",
    "griewank",
    "holder",
    "levy",
    "levy13",
    "rastr",
)

# Number of arguments for each function
myNargs = {}
myNargs["branin"] = 2
myNargs["hart3"] = 3
myNargs["hart6"] = 6
myNargs["shekel"] = 4
myNargs["ackley"] = 15
myNargs["levy"] = 20
myNargs["powell"] = 24
myNargs["michal"] = 25
myNargs["spheref"] = 27
myNargs["rastr"] = 30
myNargs["mccorm"] = 2
myNargs["bukin6"] = 2
myNargs["camel6"] = 2
myNargs["crossit"] = 2
myNargs["drop"] = 2
myNargs["egg"] = 2
myNargs["griewank"] = 2
myNargs["holder"] = 2
myNargs["levy"] = 4
myNargs["levy13"] = 2
myNargs["rastr"] = 4

# Algorithms to be tested
algorithms = {}
algorithms["SRS"] = {
    "model": rbf.RbfModel(rbf.RbfKernel.CUBIC, filter=rbf.MedianLpfFilter()),
    "optimizer": optimize.multistart_msrs,
    "acquisition": None,
}
algorithms["DYCORS"] = {
    "model": rbf.RbfModel(rbf.RbfKernel.CUBIC, filter=rbf.MedianLpfFilter()),
    "optimizer": optimize.dycors,
    "acquisition": None,
}
algorithms["CPTV"] = {
    "model": rbf.RbfModel(rbf.RbfKernel.CUBIC, filter=rbf.MedianLpfFilter()),
    "optimizer": optimize.cptv,
    "acquisition": None,
}
algorithms["CPTVl"] = {
    "model": rbf.RbfModel(rbf.RbfKernel.CUBIC, filter=rbf.MedianLpfFilter()),
    "optimizer": optimize.cptvl,
    "acquisition": None,
}
algorithms["MLSL"] = {
    "model": rbf.RbfModel(rbf.RbfKernel.CUBIC, filter=rbf.MedianLpfFilter()),
    "optimizer": optimize.surrogate_optimization,
    "acquisition": acquisition.MinimizeSurrogate(1, 0.005 * np.sqrt(2.0)),
}
algorithms["GP"] = {
    "model": gp.GaussianProcess(
        kernel=GPkernelRBF(), n_restarts_optimizer=20, normalize_y=True
    ),
    "optimizer": optimize.bayesian_optimization,
    "acquisition": acquisition.MaximizeEI(),
}

# Maximum number of evaluations
maxEvals = {}  # [20*n for n in myNargs]
for rfunc in myRfuncs:
    maxEvals[rfunc] = 100 * (myNargs[rfunc] + 1)

if __name__ == "__main__":
    import argparse

    # Set seeds for reproducibility
    np.random.seed(3)

    parser = argparse.ArgumentParser(
        description="Run given algorithm and problem from the vlse benchmark"
    )
    parser.add_argument(
        "-a", "--algorithm", choices=algorithms.keys(), default="CPTVl"
    )
    parser.add_argument("-p", "--problem", choices=myRfuncs, default="branin")
    parser.add_argument("-n", "--ntrials", type=int, default=3)
    parser.add_argument(
        "-b",
        "--bounds",
        metavar="[low,high]",
        type=float,
        nargs="+",
        help="Pass in order: low0, high0, low1, high1, ...",
    )
    args = parser.parse_args()

    # Process bounds
    if args.bounds is not None:
        bounds = [
            [args.bounds[2 * i], args.bounds[2 * i + 1]]
            for i in range(len(args.bounds) // 2)
        ]
    else:
        bounds = None

    # Print params
    print(args.algorithm)
    print(args.problem)
    print(bounds)
    print(args.ntrials)

    t0 = time.time()
    optres = run_optimizer(
        args.problem,
        myNargs[args.problem],
        maxEvals[args.problem],
        algorithms[args.algorithm],
        args.ntrials,
        bounds=bounds,
    )
    tf = time.time()

    # Save the results
    folder = os.path.dirname(os.path.abspath(__file__)) + "/pickle"
    Path(folder).mkdir(parents=True, exist_ok=True)
    filepath = (
        folder
        + "/"
        + args.problem
        + "_"
        + args.algorithm
        + "_"
        + ("bounds" if bounds else "default")
        + ".pkl"
    )
    with open(filepath, "wb") as f:
        pickle.dump(
            [
                myNargs[args.problem],
                maxEvals[args.problem],
                args.ntrials,
                optres,
                (tf - t0),
                bounds,
            ],
            f,
        )
