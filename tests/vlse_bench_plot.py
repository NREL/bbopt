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
__version__ = "0.2.0"
__deprecated__ = False

import os
import numpy as np
import pickle
import time
from random import seed
from test_vlse_bench import run_optimizer
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
# algorithms["MLSL"] = {
#     "optimizer": optimize.target_value_optimization,
#     "acquisition": acquisition.MinimizeSurrogate(1, 0.005 * np.sqrt(2.0)),
# }

# Maximum number of evaluations
maxEvals = {}  # [20*n for n in myNargs]
for rfunc in myRfuncs:
    maxEvals[rfunc] = 100 * (myNargs[rfunc] + 1)

if __name__ == "__main__":
    import argparse

    # Set seeds for reproducibility
    seed(3)
    np.random.seed(3)

    parser = argparse.ArgumentParser(
        description="Run given algorithm and problem from the vlse benchmark"
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        choices=algorithms.keys(),
        default="CPTVl",
    )
    parser.add_argument(
        "-p",
        "--problem",
        choices=myRfuncs,
        default="branin",
    )
    parser.add_argument(
        "-n",
        "--ntrials",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    print(args.algorithm)
    print(args.problem)
    print(args.ntrials)
    fulldir = os.path.dirname(os.path.abspath(__file__))

    t0 = time.time()
    optres = run_optimizer(
        args.problem,
        myNargs[args.problem],
        maxEvals[args.problem],
        algorithms[args.algorithm],
        args.ntrials,
    )
    tf = time.time()
    # Save the results
    with open(
        fulldir
        + "/pickle/vlse_bench_plot_"
        + args.problem
        + "_"
        + args.algorithm
        + ".pkl",
        "wb",
    ) as f:
        pickle.dump(
            [
                myNargs[args.problem],
                maxEvals[args.problem],
                args.ntrials,
                optres,
                (tf - t0),
            ],
            f,
        )
