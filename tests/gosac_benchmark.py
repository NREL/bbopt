import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class Problem:
    """A class to represent a problem for the GOSAC benchmark.

    Attributes
    ----------
    objf : Callable[[np.ndarray], np.ndarray]
        The objective function. Receives a 2D array of shape (n, dim) and
        returns a 1D array of shape (n,).
    gfun : Callable[[np.ndarray], np.ndarray]
        The constraint function. Receives a 2D array of shape (n, dim) and
        returns a 2D array of shape (n, gdim).
    iindex : tuple[int, ...]
        The indices of the integer variables.
    bounds : tuple[tuple[float, float], ...]
        The bounds of the variables.
    xmin : tuple[float, ...] | None
        The known minimum of the objective function. If None, the minimum is
        unknown.
    fmin : float | None
        The value of the objective function at the known minimum. If None, the
        minimum is unknown.
    """

    objf: Callable[[np.ndarray], np.ndarray]
    gfun: Callable[[np.ndarray], np.ndarray]
    iindex: tuple[int, ...]
    bounds: tuple[tuple[float, float], ...]
    xmin: tuple[float, ...] | None = None
    fmin: float | None = None


# Problems from the GOSAC benchmark
gosac_p: list[Problem] = []
gosac_p.append(
    Problem(
        # f(x) = 5(x2 − 0.2)2 + 0.8 − 0.7x1
        lambda x: 5 * (x[:, 1] - 0.2) ** 2 + 0.8 - 0.7 * x[:, 0],
        # −exp(x2 −0.2)−x3 ≤ 0
        # x3 + 1.1x1 + 1 ≤ 0
        # x2 − 1.2x1 ≤ 0
        lambda x: np.transpose(
            [
                -np.exp(x[:, 1] - 0.2) - x[:, 2],
                x[:, 2] + 1.1 * x[:, 0] + 1,
                x[:, 1] - 1.2 * x[:, 0],
            ]
        ),
        (0,),
        ((0.0, 1.0), (0.2, 1.0), (-2.22554, -1.0)),
        (1, 0.9419, -2.1),
        2.8524,
    )
)
gosac_p.append(
    Problem(
        # 2x1 + 3x2 + 1.5x3 + 2x4 − 0.5x5
        lambda x: 2 * x[:, 0]
        + 3 * x[:, 1]
        + 1.5 * x[:, 2]
        + 2 * x[:, 3]
        - 0.5 * x[:, 4],
        # x21 + x3 ≤ 1.25
        # x1.5 + 1.5x4 ≤ 3
        # x1 + x3 ≤ 1.6
        # 1.333x2 + x4 ≤ 3
        # −x3 − x4 + x5 ≤ 0
        lambda x: np.transpose(
            [
                x[:, 0] ** 2 + x[:, 2] - 1.25,
                x[:, 1] ** 1.5 + 1.5 * x[:, 3] - 3,
                x[:, 0] + x[:, 2] - 1.6,
                1.333 * x[:, 1] + x[:, 3] - 3,
                -x[:, 2] - x[:, 3] + x[:, 4],
            ]
        ),
        (0, 1),
        ((0, 10), (0, 10), (0, 10), (0, 1), (0, 1)),
        (0, 0, 0, 0, 0),
        0,
    )
)
gosac_p.append(
    Problem(
        # f(x) = 2x1 + 3x2 + 1.5x3 + 2x4 − 0.5x5
        lambda x: 2 * x[:, 0]
        + 3 * x[:, 1]
        + 1.5 * x[:, 2]
        + 2 * x[:, 3]
        - 0.5 * x[:, 4],
        # x1 + x3 ≤ 1.6
        # 1.333x2 + x4 ≤ 3
        # −x3 −x4 +x5 ≤0
        lambda x: np.transpose(
            [
                x[:, 0] + x[:, 2] - 1.6,
                1.333 * x[:, 1] + x[:, 3] - 3,
                -x[:, 2] - x[:, 3] + x[:, 4],
            ]
        ),
        (0, 1, 2),
        ((0, 10), (0, 10), (0, 10), (0, 1), (0, 1)),
        (0, 0, 0, 0, 0),
        0,
    )
)
gosac_p.append(
    Problem(
        # f(x) = (x5 −1)2 +(x6 −2)2 +(x7 −1)2 −log(x8 +1) +(x9 −1)2 +(x10 −2)2 +(x11 −3)2
        lambda x: (x[:, 4] - 1) ** 2
        + (x[:, 5] - 2) ** 2
        + (x[:, 6] - 1) ** 2
        - np.log(x[:, 7] + 1)
        + (x[:, 8] - 1) ** 2
        + (x[:, 9] - 2) ** 2
        + (x[:, 10] - 3) ** 2,
        lambda x: np.transpose(
            [
                x[:, 0] + x[:, 1] + x[:, 2] + x[:, 8] + x[:, 9] + x[:, 10] - 5,
                x[:, 6] ** 2
                + x[:, 8] ** 2
                + x[:, 9] ** 2
                + x[:, 10] ** 2
                - 5.5,
                x[:, 0] + x[:, 8] - 1.2,
                x[:, 1] + x[:, 9] - 1.8,
                x[:, 2] + x[:, 10] - 2.5,
                x[:, 3] + x[:, 8] - 1.2,
                x[:, 5] ** 2 + x[:, 9] ** 2 - 1.64,
                x[:, 6] ** 2 + x[:, 10] ** 2 - 4.25,
                x[:, 5] ** 2 + x[:, 10] ** 2 - 4.64,
                x[:, 4] - x[:, 0],
                x[:, 5] - x[:, 1],
                x[:, 6] - x[:, 2],
                x[:, 7] - x[:, 3],
            ]
        ),
        (0, 1, 2, 3),
        (
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 10),
            (0, 10),
            (0, 10),
        ),
        (1, 1, 0, 1, 1, 1, 0, 1, 0.2, 0.8, 1.9079),
        4.5796,
    )
)
gosac_p.append(
    Problem(
        lambda x: -np.abs(
            (
                np.sum(np.cos(x) ** 4, axis=1)
                - 2 * np.prod(np.cos(x) ** 2, axis=1)
            )
            / np.sqrt(np.sum(np.arange(1, x.shape[1] + 1) * x**2, axis=1))
        ),
        lambda x: np.transpose(
            [
                0.75 - np.prod(x, axis=1),
                np.sum(x, axis=1) - 7.5 * x.shape[1],
            ]
        ),
        (0, 1, 2, 3, 4, 5),
        ((0, 10),) * 25,
        (
            3,
            3,
            3,
            3,
            3,
            3,
            3.021616897,
            2.998077169,
            0.3770080798,
            0.3780122315,
            0.3737440086,
            2.900804325,
            2.871928856,
            2.843098301,
            2.806423432,
            0.3625403532,
            0.3622042429,
            0.3593120409,
            0.3589796805,
            0.3556172311,
            0.3542916697,
            0.3534646982,
            0.3511441992,
            0.3492637643,
            0.3467394526,
        ),
        -0.73904,
    )
)
gosac_p.append(
    Problem(
        # 5.3578547x23 + 0.8356891x1 x5 + 37.293239x1 − 40792.141
        lambda x: 5.3578547 * x[:, 2] ** 2
        + 0.8356891 * x[:, 0] * x[:, 4]
        + 37.293239 * x[:, 0]
        - 40792.141,
        # 0 ≤ 85.334407 + 0.0056858x2x5 + 0.0006262x1x4 − 0.0022053x3x5 ≤ 92
        # 90 ≤ 80.51249 + 0.0071317x2x5 + 0.0029955x1x2 + 0.0021813x23 ≤ 110
        # 20 ≤ 9.300961 + 0.0047026x3x5 + 0.0012547x1x3 + 0.0019085x3x4 ≤ 25
        lambda x: np.transpose(
            [
                -85.334407
                - 0.0056858 * x[:, 1] * x[:, 4]
                - 0.0006262 * x[:, 0] * x[:, 3]
                + 0.0022053 * x[:, 2] * x[:, 4],
                85.334407
                + 0.0056858 * x[:, 1] * x[:, 4]
                + 0.0006262 * x[:, 0] * x[:, 3]
                - 0.0022053 * x[:, 2] * x[:, 4]
                - 92,
                90
                - 80.51249
                - 0.0071317 * x[:, 1] * x[:, 4]
                - 0.0029955 * x[:, 0] * x[:, 1]
                - 0.0021813 * x[:, 2] ** 2,
                80.51249
                + 0.0071317 * x[:, 1] * x[:, 4]
                + 0.0029955 * x[:, 0] * x[:, 1]
                + 0.0021813 * x[:, 2] ** 2
                - 110,
                20
                - 9.300961
                - 0.0047026 * x[:, 2] * x[:, 4]
                - 0.0012547 * x[:, 0] * x[:, 2]
                - 0.0019085 * x[:, 2] * x[:, 3],
                9.300961
                + 0.0047026 * x[:, 2] * x[:, 4]
                + 0.0012547 * x[:, 0] * x[:, 2]
                + 0.0019085 * x[:, 2] * x[:, 3]
                - 25,
            ]
        ),
        (0, 1),
        ((78, 102), (33, 45), (27, 45), (27, 45), (27, 45)),
        (78, 33, 29.9953, 45, 36.7758),
        -30665.54,
    )
)
gosac_p.append(
    Problem(
        # (x1 −10)2 +5(x2 −12)2 +x43 +3(x4 −11)2+10x65 + 7x26 + x47 − 4x6x7 − 10x6 − 8x7
        lambda x: (x[:, 0] - 10) ** 2
        + 5 * (x[:, 1] - 12) ** 2
        + x[:, 3] ** 2
        + 3 * (x[:, 3] - 11) ** 2
        + 10 * x[:, 5] ** 2
        + 7 * x[:, 2] ** 2
        + x[:, 4] ** 2
        - 4 * x[:, 2] * x[:, 3]
        - 10 * x[:, 2]
        - 8 * x[:, 3],
        # 2x21 +3x42 +x3 +4x24 +5x5 ≤ 127
        # 7x1 +3x2 +10x23 +x4 −x5 ≤ 282
        # 23x1 +x2 +6x26 −8x7 ≤196
        # 4x21 +x2 −3x1x2 +2x23 +5x6 −11x7 ≤ 0
        lambda x: np.transpose(
            [
                2 * x[:, 0] ** 2
                + 3 * x[:, 1] ** 4
                + x[:, 2]
                + 4 * x[:, 3] ** 2
                + 5 * x[:, 4]
                - 127,
                7 * x[:, 0]
                + 3 * x[:, 1]
                + 10 * x[:, 2] ** 2
                + x[:, 3]
                - x[:, 4]
                - 282,
                23 * x[:, 0]
                + x[:, 1] ** 2
                + 6 * x[:, 5] ** 2
                - 8 * x[:, 6]
                - 196,
                4 * x[:, 0] ** 2
                + x[:, 1] ** 2
                - 3 * x[:, 0] * x[:, 1]
                + 2 * x[:, 2] ** 2
                + 5 * x[:, 5]
                - 11 * x[:, 6],
            ]
        ),
        (0, 1, 2),
        ((-10, 10),) * 7,
        (
            2,
            2,
            -1,
            4.32299345363447,
            -0.550622364659113,
            1.16531693494537,
            1.46201272171894,
        ),
        682.936407277722,
    )
)
