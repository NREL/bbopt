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


def SLHDstandard(d, m):
    """SLHD creates a symmetric latin hypercube design. d is the dimension of the input and

    m is the number of initial points to be selected.
    """
    delta = (1.0 / m) * np.ones(d)
    X = np.zeros([m, d])
    for j in range(d):
        for i in range(m):
            X[i, j] = ((2.0 * (i + 1) - 1) / 2.0) * delta[j]
    P = np.zeros([m, d], dtype=int)
    P[:, 0] = np.arange(m)
    if m % 2 == 0:
        k = m // 2
    else:
        k = (m - 1) // 2
        P[k, :] = (k + 1) * np.ones((1, d))

    for j in range(1, d):
        P[0:k, j] = np.random.permutation(np.arange(k))
        for i in range(k):
            if np.random.random() < 0.5:
                P[m - 1 - i, j] = m - 1 - P[i, j]
            else:
                P[m - 1 - i, j] = P[i, j]
                P[i, j] = m - 1 - P[i, j]
    InitialPoints = np.zeros([m, d])
    for j in range(d):
        for i in range(m):
            InitialPoints[i, j] = X[P[i, j], j]
    return InitialPoints


if __name__ == "__main__":
    print("This is test for SLHDstandard")
    dim = 3
    m = 2 * (dim + 1)
    print("dim is", dim)
    print("m is", m)
    print("set seed to 5")
    np.random.seed(5)
    for i in range(3):
        print(SLHDstandard(dim, m))
