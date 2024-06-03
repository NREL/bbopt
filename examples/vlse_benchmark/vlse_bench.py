"""Run the optimization on the VLSE benchmark."""

# Copyright (C) 2024 National Renewable Energy Laboratory

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
__version__ = "0.3.3"
__deprecated__ = False

import os
import numpy as np
import pickle
import time
from tests.test_vlse_bench import run_optimizer
from blackboxopt import optimize, acquisition, sampling

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
    "optimizer": optimize.multistart_stochastic_response_surface,
    "acquisition": acquisition.CoordinatePerturbation(
        0,
        sampling.NormalSampler(
            1,
            sigma=0.2,
            sigma_min=0.2 * 0.5**5,
            sigma_max=0.2,
            strategy=sampling.SamplingStrategy.NORMAL,
        ),
        [0.3, 0.5, 0.8, 0.95],
    ),
}
algorithms["DYCORS"] = {
    "optimizer": optimize.multistart_stochastic_response_surface,
    "acquisition": acquisition.CoordinatePerturbation(
        0,
        sampling.NormalSampler(
            1,
            sigma=0.2,
            sigma_min=0.2 * 0.5**5,
            sigma_max=0.2,
            strategy=sampling.SamplingStrategy.DDS,
        ),
        [0.3, 0.5, 0.8, 0.95],
    ),
}
algorithms["CPTV"] = {
    "optimizer": optimize.cptv,
    "acquisition": acquisition.CoordinatePerturbation(
        0,
        sampling.NormalSampler(
            1,
            sigma=0.2,
            sigma_min=0.2 * 0.5**5,
            sigma_max=0.2,
            strategy=sampling.SamplingStrategy.DDS,
        ),
        [0.3, 0.5, 0.8, 0.95],
    ),
}
algorithms["CPTVl"] = {
    "optimizer": optimize.cptvl,
    "acquisition": acquisition.CoordinatePerturbation(
        0,
        sampling.NormalSampler(
            1,
            sigma=0.2,
            sigma_min=0.2 * 0.5**5,
            sigma_max=0.2,
            strategy=sampling.SamplingStrategy.DDS,
        ),
        [0.3, 0.5, 0.8, 0.95],
    ),
}
algorithms["MLSL"] = {
    "optimizer": optimize.target_value_optimization,
    "acquisition": acquisition.MinimizeSurrogate(1, 0.005 * np.sqrt(2.0)),
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
    filepath = (
        os.path.dirname(os.path.abspath(__file__))
        + "/pickle/vlse_bench_plot_"
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
