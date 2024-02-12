def welchetal92(xx):
    # Extracting values from input vector xx
    (
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        x8,
        x9,
        x10,
        x11,
        x12,
        x13,
        x14,
        x15,
        x16,
        x17,
        x18,
        x19,
        x20,
    ) = xx

    # Calculating each term of the function
    term1 = 5 * x12 / (1 + x1)
    term2 = 5 * (x4 - x20) ** 2
    term3 = x5 + 40 * x19**3 - 5 * x19
    term4 = 0.05 * x2 + 0.08 * x3 - 0.03 * x6
    term5 = 0.03 * x7 - 0.09 * x9 - 0.01 * x10
    term6 = -0.07 * x11 + 0.25 * x13**2 - 0.04 * x14
    term7 = 0.06 * x15 - 0.01 * x17 - 0.03 * x18

    # Summing up all terms to compute the final result
    y = term1 + term2 + term3 + term4 + term5 + term6 + term7
    return y
