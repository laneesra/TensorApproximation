import numpy as np
import time
from scipy.linalg import inv
import math

from tensor import Tensor


def bunch_kaufman(A):
    def pivot_1(k, n):
        c = A[k + 1:, k]
        L[k + 1:, k] = c / A[k, k]
        A[k + 1:, k + 1:] -= np.dot(c, c.T) / A[k, k]
        A[k + 1:, k:k + 1] = np.zeros((n - (k + 1), 1))
        A[k:k + 1, k + 1:] = np.zeros((1, n - (k + 1)))
        pivot[k] = 1

    def pivot_2(k, n):
        E = [[A[k, k], A[k + 1, k]], [A[k + 1, k], A[k + 1, k + 1]]]
        E_inv = inv(E)

        c = A[k + 2:n, k:k + 2]
        L[k + 2:, k:k + 2] = np.dot(c, E_inv)
        A[k + 2:, k + 2:] -= np.dot(np.dot(c, E_inv), c.T)
        A[k + 2:, k:k + 2] = np.zeros((n - (k + 2), 2))
        A[k:k + 2, k + 2:] = np.zeros((2, n - (k + 2)))
        pivot[k] = 2

    n = A.shape[0]
    L = np.matrix(np.eye(n, n), dtype=np.double)
    alpha = (1. + np.sqrt(17)) / 8
    pivot = np.zeros(n, dtype=int)
    P = np.arange(0, n, 1)
    k = 0

    while k < n:
        lambda_1, j = np.max(np.abs(A[k:, k])), np.argmax(np.abs(A[k:, k]))
        r = k + j

        if np.abs(A[k, k]) >= alpha * lambda_1: # 1x1 pivot
            pivot_1(k, n)
            k += 1

        else:
            if r + 1 == n:
                lambda_r = np.max(np.abs(A[r, k:r]))
            else:
                lambda_r = np.max([np.max(np.abs(A[r, k:r])), np.max(np.abs(A[r + 1:, r]))])

            if np.abs(A[k, k]) * lambda_r >= alpha * lambda_1 * lambda_1: # 1x1 pivot
                pivot_1(k, n)
                k += 1

            else:
                if np.abs(A[r, r]) >= alpha * lambda_r: # 1x1 pivot
                    P[k], P[r] = P[r], P[k]
                    A[k, k], A[r, r] = A[r, r], A[k, k]
                    A[r + 1:, r], A[r + 1:, k] = A[r + 1:, k], A[r + 1:, r].copy()
                    A[k + 1:r, k], A[r, k + 1:r] = A[r, k + 1:r].T, A[k + 1:r, k].T
                    L[k, :k], L[r, :k] = L[r, :k], L[k, :k].copy()

                    pivot_1(k, n)
                    k += 1

                else:
                    if np.abs(A[r, r]) < alpha * lambda_r: # 2x2 pivot
                        P[k + 1], P[r] = P[r], P[k + 1]
                        A[k + 1, k + 1], A[r, r] = A[r, r], A[k + 1, k + 1]
                        A[r + 1:, k + 1], A[r + 1:, r] = A[r + 1:, r], A[r + 1:, k + 1].copy()
                        A[k + 1, k], A[r, k] = A[r, k], A[k + 1, k]
                        A[k + 2:r, k + 1], A[r, k + 2:r] = A[r, k + 2:r].T, A[k + 2:r, k + 1].T

                        L[k + 1, :k], L[r, :k] = L[r, :k], L[k + 1, :k].copy()
                        pivot_2(k, n)
                        k += 2

    P_ = np.zeros(L.shape)
    for i in range(len(P_)):
        P_[i][P[i]] = 1.

    return A, L, P_, pivot


def solve(A):
    D, L, P, pivot = bunch_kaufman(A)
    return D, L, P


def find_lambdas(D, pivot):
    k = 0
    lambdas = []
    for p in pivot:
        print('u', np.linalg.eig(D[k:k + p, k:k + p])[1])
        lambdas += list(np.linalg.eig(D[k:k + p, k:k + p])[0])
        k += p
    return lambdas


def sort_by_permutation(d):
    ds = [(d[i], i) for i in range(len(d))]
    ds.sort(key=lambda x: abs(x[0]), reverse=True)
    perm = [(ds[i][1], i) for i in range(len(d))]
    P = np.zeros((len(d), len(d)))
    for p1, p2 in perm:
        P[p2][p1] = 1
    return P

if __name__ == '__main__':
    A = np.matrix([[6, 2, 3, -6, 1], [5, 6, 2, -1, 2], [-1, 1, 4, 3, -7], [2, 5, 3, -1, 6]], dtype=float)
    #A = np.matrix([[6, 12, 3, -6], [12, -8 ,-13, 4], [3, -13, -7, 1], [-6, 4, 1, 6]], dtype=float)
    S = np.dot(A.H, A)
    print('S', S)
    print('shape', A.shape)
    start = time.time()
    D, L, P, pivot = bunch_kaufman(S)
    r = np.linalg.matrix_rank(A)
    print(r)
    print('BK in {} secs\n'.format(time.time() - start))
    S_bk = P @ L @ D @ L.H @ P.T
    print('S_bk', S_bk)
    print('D', D)
    print('L', L)
    print('P', P)
    lambdas, U = np.linalg.eig(D)
    #Q = sort_by_permutation(lambdas)
    print('lambdas', lambdas)
    l1 = [1. / math.sqrt(lambdas[i]) if i < r else 0 for i in range(len(lambdas)) ]
    #print('Q', Q)
    #print('res', np.dot(np.dot(Q, np.diag(lambdas)), Q.T))
    B = A @ P @ inv(L.H) @ U @ np.diag(l1)
    B = B[:, :r]
    lambdas, U = lambdas[:r], U[:r]

    print('B', B)
    print('Er', B.H @ B)
    V = np.matrix(np.random.rand(A.shape[0], A.shape[0]))
    V[:B.shape[0], :B.shape[1]] = B
    print('V', V @ V.H)

    C = np.matrix(np.random.rand(A.shape[1], A.shape[1]))
    C[:U.shape[0], :U.shape[1]] = U
    sigma = np.zeros(min(A.shape))
    sigma[:r] = [math.sqrt(l) for l in lambdas]

    Q = sort_by_permutation(sigma)
    sigma = np.diag(sigma)
    W_inv = C.H @ L.H @ P.T
    Q1 = np.eye(A.shape[0], A.shape[0])
    Q2 = np.eye(A.shape[1], A.shape[1])
    Sigma = np.eye(A.shape[0], A.shape[1])

    Sigma[:r, :r] = sigma
    Q1[:Q.shape[0], :Q.shape[1]] = Q
    Q2[:Q.shape[0], :Q.shape[1]] = Q.T

#    A_ = V @ Q1.T @ (Q1 @ Sigma @ Q2) @ Q2.T @ W_inv
    rk = len(S)
    Sigma = (Q1 @ Sigma @ Q2)
    print('sigma', Sigma)
    print('svd', np.linalg.svd(A)[1])
    V = V @ Q1.T
    W_inv = Q2.T @ W_inv
    t = Tensor(A)
    for i in range(len(Sigma), 0, -1):
        A1 = V[:, :i] @ Sigma[:i, :i] @ W_inv[:i, :]
        print(A1, '\n')
        E = np.array(A - A1)
        print(np.sum(np.sum(E * E)) ** 0.5)

    print(A)

