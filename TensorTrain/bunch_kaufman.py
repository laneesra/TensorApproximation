import numpy as np
import copy


def bunch_kaufman(A):
    N = A.shape[0]
    L = np.matrix(np.eye(N, N), dtype=np.float64)
    alpha = (1 + np.sqrt(17)) / 8
    pivot = np.zeros(N)
    P = np.arange(1, N + 1, 1)
    k = 0

    while k < N:

        lambda_1, j = np.max(np.abs(A[k:N, k])), np.argmax(np.abs(A[k:N, k]))
        r = k + j

        if np.abs(A[k, k]) >= alpha * lambda_1:

            L[k + 1:N, k] = A[k + 1:N, k].copy() / A[k, k]
            A[k + 1:N, k + 1:N] = A[k + 1:N, k + 1:N] - L[k + 1:N, k] * A[k + 1:N, k].T
            A[k + 1:N, k] = np.matrix(np.zeros(N - (k + 1))).T

            pivot[k] = 1
            k = k + 1

        else:

            if r < N - 1:
                lambda_r = np.max([np.max(np.abs(A[r, k:r])), np.max(np.abs(A[r + 1:N, r]))])

            else:
                lambda_r = np.max(np.abs(A[r, k:r]))

            if np.abs(A[k, k]) * lambda_r >= alpha * lambda_1 * lambda_1:
                L[k + 1:N, k] = A[k + 1:N, k].copy() / A[k, k]
                A[k + 1:N, k + 1:N] = A[k + 1:N, k + 1:N] - L[k + 1:N, k] * A[k + 1:N, k].T
                A[k + 1:N, k] = np.matrix(np.zeros(N - (k + 1))).T

                pivot[k] = 1
                k = k + 1

            else:

                if np.abs(A[r, r]) >= alpha * lambda_r:
                    P[k], P[r] = P[r], P[k].copy()
                    A[k, k], A[r, r] = A[r, r], A[k, k].copy()
                    A[r + 1:N, r], A[r + 1:N, k] = A[r + 1:N, k], A[r + 1:N, r].copy()
                    A[k + 1:r, k], A[r, k + 1:r] = np.transpose(A[r, k + 1:r]), np.transpose(A[k + 1:r, k].copy())

                    if k > 0:
                        L[k, 0:k], L[r, 0:k] = L[r, 0:k], L[k, 0:k].copy()

                    L[k + 1:N, k] = A[k + 1:N, k] / A[k, k]
                    A[k + 1:N, k + 1:N] = A[k + 1:N, k + 1:N] - L[k + 1:N, k] * A[k + 1:N, k].T
                    A[k + 1:N, k] = np.matrix(np.zeros(N - (k + 1))).T

                    pivot[k] = 1
                    k = k + 1

                else:
                    if np.abs(A[r, r]) < alpha * lambda_r:
                        P[k + 1], P[r] = P[r], P[k + 1].copy()
                        A[k + 1, k + 1], A[r, r] = A[r, r], A[k + 1, k + 1].copy()
                        A[r + 1:N, k + 1], A[r + 1:N, r] = A[r + 1:N, r], A[r + 1:N, k + 1].copy()
                        A[k + 1, k], A[r, k] = A[r, k], A[k + 1, k].copy()
                        A[k + 2:r, k + 1], A[r, k + 2:r] = np.transpose(A[r, k + 2:r]), np.transpose(
                            A[k + 2:r, k + 1].copy())

                        if k > 0:
                            L[k + 1, 1:k], L[r, 1:k] = L[r, 1:k], L[k + 1, 1:k].copy()

                        E = np.eye(2, 2)

                        E[0, 0] = A[k, k]
                        E[1, 0] = A[k + 1, k]
                        E[0, 1] = E[1, 0]
                        E[1, 1] = A[k + 1, k + 1]

                        detE = E[0, 0] * E[1, 1] - E[0, 1] * E[1, 0]
                        invE = np.array([[E[1, 1], -E[1, 0]], [-E[1, 0], E[0, 0]]]) / detE

                        L[k + 2:N, k:k + 2] = np.matmul(A[k + 2:N, k:k + 2].copy(), invE)
                        A[k + 2:N, k + 2:N] = A[k + 2:N, k + 2:N] - L[k + 2:N, k:k + 2] * A[k + 2:N, k:k + 2].T
                        A[k + 2:N, k] = np.matrix(np.zeros(N - (k + 2))).T
                        A[k + 2:N, k + 1] = np.matrix(np.zeros(N - (k + 2))).T

                        pivot[k] = 2
                        k = k + 2

    if 0 == pivot[N - 1]:
        if 1 == pivot[N - 2]:
            pivot[N - 1] = 1

    for ind in range(0, N - 1):
        A[ind, ind + 1:] = 0.0

    return A, L, P - 1, pivot


def solve(A):
    D, L, P, pivot = bunch_kaufman(copy.deepcopy(A))
    '''print('D', D)
    print('P', P)
    print('L', L)
    print('pivot', pivot'''
    return D, L, P


if __name__ == '__main__':

    A = np.matrix([[1, 10, 20],[10, 1 ,30],[20, 30, 1]])
    solve(A)