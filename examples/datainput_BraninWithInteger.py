"""Data class for the Branin problem definition.
"""

# Copyright (C) 2024 National Renewable Energy Laboratory
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


def datainput_BraninWithInteger():
    return Data(
        xlow=np.array([-5, 0]),
        xup=np.array([10, 15]),
        objfunction=myfun,
        dim=2,
        iindex=(0,),
    )


def myfun(x):
    assert x.size == 2
    xflat = x.flatten()
    y = (
        pow(
            xflat[1]
            - 5.1 * pow(xflat[0], 2) / (4 * pow(np.pi, 2))
            + 5 * xflat[0] / np.pi
            - 6,
            2,
        )
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(xflat[0])
        + 10
    )
    return y


if __name__ == "__main__":
    print(myfun(np.array([[3, 2.4013]])))
    print(myfun(np.array([[3, 2.26325204]])))
