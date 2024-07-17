from enum import Enum
from types import MappingProxyType
import numpy as np

RbfKernel = Enum(
    "RbfKernel",
    [
        "LINEAR",
        "CUBIC",
        "THINPLATE",
        "QUINTIC",
        "MULTIQUADRIC",
        "INVERSE_MULTIQUADRIC",
        "INVERSE_QUADRATIC",
        "GAUSSIAN",
    ],
)

# Kernel functions


def linear(r):
    return np.negative(r)


def thin_plate_spline(r):
    if not hasattr(r, "__len__"):
        ret = 1 if r == 0 else r
    else:
        ret = np.where(np.asarray(r) == 0, 1, r)
    return ret**2 * np.log(ret)


def cubic(r):
    return np.power(r, 3)


def quintic(r):
    return -np.power(r, 5)


def multiquadric(r):
    return -np.sqrt(np.power(r, 2) + 1)


def inverse_multiquadric(r):
    return 1 / np.sqrt(np.power(r, 2) + 1)


def inverse_quadratic(r):
    return 1 / (np.power(r, 2) + 1)


def gaussian(r):
    return np.exp(-np.power(r, 2))


# Kernel derivatives


def d_linear(r: np.ndarray):
    return -np.ones(r.shape)


def d_thin_plate_spline(r: np.ndarray):
    ret = np.where(r == 0, 1, r)
    return 2 * ret * np.log(ret) + ret


def d_cubic(r: np.ndarray):
    return 3 * r**2


def d_quintic(r: np.ndarray):
    return -5 * r**4


def d_multiquadric(r: np.ndarray):
    return -r / np.sqrt(r**2 + 1)


def d_inverse_multiquadric(r: np.ndarray):
    return -r / np.sqrt(r**2 + 1) ** 3


def d_inverse_quadratic(r: np.ndarray):
    return -2 * r / (r**2 + 1) ** 2


def d_gaussian(r: np.ndarray):
    return -2 * r * np.exp(-(r**2))


# Kernel derivatives over r


def d_linear_over_r(r: np.ndarray):
    return -1 / r


def d_thin_plate_spline_over_r(r: np.ndarray):
    return 2 * np.log(r) + 1


def d_cubic_over_r(r: np.ndarray):
    return 3 * r


def d_quintic_over_r(r: np.ndarray):
    return -5 * r**3


def d_multiquadric_over_r(r: np.ndarray):
    return -1 / np.sqrt(r**2 + 1)


def d_inverse_multiquadric_over_r(r: np.ndarray):
    return -1 / np.sqrt(r**2 + 1) ** 3


def d_inverse_quadratic_over_r(r: np.ndarray):
    return -2 / (r**2 + 1) ** 2


# Kernel second derivatives


def d2_linear(r: np.ndarray):
    return np.zeros(r.shape)


def d2_thin_plate_spline(r: np.ndarray):
    ret = np.where(r == 0, 1, r)
    return 2 * np.log(ret) + 3


def d2_cubic(r: np.ndarray):
    return 6 * r


def d2_quintic(r: np.ndarray):
    return -20 * r**3


def d2_multiquadric(r: np.ndarray):
    return -1 / np.sqrt(r**2 + 1) ** 3


def d2_inverse_multiquadric(r: np.ndarray):
    return (2 * r**2 - 1) / np.sqrt(r**2 + 1) ** 5


def d2_inverse_quadratic(r: np.ndarray):
    return (6 * r**2 - 2) / (r**2 + 1) ** 3


def d2_gaussian(r: np.ndarray):
    return (4 * r**2 - 2) * np.exp(-(r**2))


KERNEL_FUNC = MappingProxyType(
    {
        RbfKernel.LINEAR: linear,
        RbfKernel.CUBIC: cubic,
        RbfKernel.THINPLATE: thin_plate_spline,
        RbfKernel.QUINTIC: quintic,
        RbfKernel.MULTIQUADRIC: multiquadric,
        RbfKernel.INVERSE_MULTIQUADRIC: inverse_multiquadric,
        RbfKernel.INVERSE_QUADRATIC: inverse_quadratic,
        RbfKernel.GAUSSIAN: gaussian,
    }
)

KERNEL_DERIVATIVE_FUNC = MappingProxyType(
    {
        RbfKernel.LINEAR: d_linear,
        RbfKernel.CUBIC: d_cubic,
        RbfKernel.THINPLATE: d_thin_plate_spline,
        RbfKernel.QUINTIC: d_quintic,
        RbfKernel.MULTIQUADRIC: d_multiquadric,
        RbfKernel.INVERSE_MULTIQUADRIC: d_inverse_multiquadric,
        RbfKernel.INVERSE_QUADRATIC: d_inverse_quadratic,
        RbfKernel.GAUSSIAN: d_gaussian,
    }
)

KERNEL_DERIVATIVE_OVER_R_FUNC = MappingProxyType(
    {
        RbfKernel.LINEAR: d_linear_over_r,
        RbfKernel.CUBIC: d_cubic_over_r,
        RbfKernel.THINPLATE: d_thin_plate_spline_over_r,
        RbfKernel.QUINTIC: d_quintic_over_r,
        RbfKernel.MULTIQUADRIC: d_multiquadric_over_r,
        RbfKernel.INVERSE_MULTIQUADRIC: d_inverse_multiquadric_over_r,
        RbfKernel.INVERSE_QUADRATIC: d_inverse_quadratic_over_r,
        RbfKernel.GAUSSIAN: d_gaussian,
    }
)

KERNEL_SECOND_DERIVATIVE_FUNC = MappingProxyType(
    {
        RbfKernel.LINEAR: d2_linear,
        RbfKernel.CUBIC: d2_cubic,
        RbfKernel.THINPLATE: d2_thin_plate_spline,
        RbfKernel.QUINTIC: d2_quintic,
        RbfKernel.MULTIQUADRIC: d2_multiquadric,
        RbfKernel.INVERSE_MULTIQUADRIC: d2_inverse_multiquadric,
        RbfKernel.INVERSE_QUADRATIC: d2_inverse_quadratic,
        RbfKernel.GAUSSIAN: d2_gaussian,
    }
)
