import numpy as np


def hart6d(xx):
    # xx = [x1, x2, x3, x4, x5, x6]
    alpha = [1.0, 1.2, 3.0, 3.2]

    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])

    pm = np.array([[1312, 1696, 5569, 124, 8283, 5886],
                   [2329, 4135, 8307, 3736, 1004, 9991],
                   [2348, 1451, 3522, 2883, 3047, 6650],
                   [4047, 8828, 8732, 5743, 1091, 381]])
    P = np.power(10.0, -4.0) * pm
    outer = 0;
    for ii in range(4):
        inner = 0
        for jj in range(6):
            xj = xx[jj]
            Aij = A[ii, jj]
            Pij = P[ii, jj]
            inner = inner + Aij * np.power((xj - Pij), 2.0)
        new = alpha[ii] * np.exp(-inner)
        outer = outer + new
    y = -(2.58 + outer) / 1.94

    return y


if __name__ == "__main__":
	X = np.random.rand(5000, 6)
	y = np.apply_along_axis(hart6d, 1, X)

