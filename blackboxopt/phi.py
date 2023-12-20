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
import math
from .utility import myException


def phi(r, type):
    """determines phi-value of distance r between 2 points (depends on chosen RBF model)

    Input:
         r: distance between 2 points
         type: RBF model type

    Output:
         output: phi-value according to RBF model
    """
    if type == "linear":
        output = r
    elif type == "cubic":
        output = np.power(r, 3)
    elif type == "thinplate":
        if r >= 0:
            output = np.multiply(np.power(r, 2), math.log(r + np.finfo(np.double).tiny))
        else:
            output = np.zeros(r.shape)
    else:
        raise myException("Error: Unkonwn type.")

    return output
