"""Data class for the Rastigrin problem definition.
"""

# Copyright (C) 2023 National Renewable Energy Laboratory
# Copyright (C) 2013 Cornell University

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
from data import Data


def datainput_rastrigin():
    n = 10
    return Data(xlow=np.zeros(n), xup=np.ones(n), objfunction=myfun, dim=n)


def myfun(x):
    n = 10
    xlow = np.asarray(-10 * np.ones(n))
    xup = np.asarray(10 * np.ones(n))
    x = xlow + np.multiply(xup - xlow, x)
    y = 10 * n + sum(pow(x, 2) - 10 * np.cos(2 * np.pi * x))

    return y


if __name__ == "__main__":
    print(myfun(np.array([[0.5, 0.9]])))
