from .borehole import borehole
from .wingweight import wingweight
from .otlcircuit import otlcircuit
from .piston import piston


def moonetal12(xx):
    y = []

    # Call the functions and append the results to the list y
    y.append(borehole(xx[:8]))
    y.append(wingweight(xx[8:18]))
    y.append(otlcircuit(xx[18:24]))
    y.append(piston(xx[24:31]))

    miny = min(y)
    maxy = max(y)

    # Normalize the results
    ystar = [(yi - miny) / (maxy - miny) for yi in y]

    # Sum up the normalized results
    y = sum(ystar)
    return y
