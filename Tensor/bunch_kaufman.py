import numpy as np
import time
from scipy.linalg import inv, pinvh
from numpy.linalg import lstsq
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import matplotlib.patches as mpatches
from scipy.stats import unitary_group, ortho_group
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix
import scipy.linalg as sp


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


def find_orth(M):
    rand_vec = np.random.rand(M.shape[0], 1)
    A = np.hstack((M, rand_vec))
    b = np.zeros(M.shape[1] + 1)
    b[-1] = 1.
    l = np.array(lstsq(A.T, b)[0])
    return l


def unitary_build(V, shape):
    if V.shape[0] - shape[0] > V.shape[1] - shape[1]:
        for i in range(shape[0], V.shape[0]):
            V[i, i - shape[0]] = 1.
        if shape[1] < V.shape[1]:
            V[:, shape[1]:] = find_orth(V[:, :shape[1]])
    else:
        for i in range(shape[1], V.shape[1]):
            V[i - shape[1], i] = 1.
        if shape[0] < V.shape[0]:
            V[shape[0]:, :] = find_orth(V[:shape[0], :])
    return V


def truncated_bk(A):
    A = np.matrix(A)
    S = np.dot(A.H, A)
    D, L, P, pivot = bunch_kaufman(S.copy())
    r = np.linalg.matrix_rank(A)
    L_h = L.H

    lambdas, U = np.linalg.eig(D)
    l1 = [1. / np.sqrt(lambdas[i]) if i < r and lambdas[i] > 0 else 0 for i in range(len(lambdas))]
    B = A @ P @ inv(L_h) @ U @ np.diag(l1)
    lambdas, U = lambdas[:r], U[:r]

    V = np.matrix(np.zeros((A.shape[0], A.shape[0])))
    if B.shape[0] > V.shape[0] or B.shape[1] > V.shape[0]:
        V = B[:V.shape[0], :V.shape[1]]
    else:
        V[:B.shape[0], :B.shape[1]] = B
        V = unitary_build(V.copy(), B.shape)

    C = np.matrix(np.zeros((A.shape[1], A.shape[1])))
    if U.shape[0] > C.shape[0] or U.shape[1] > C.shape[0]:
        C = U[:C.shape[0], :C.shape[1]]
    else:
        C[:U.shape[0], :U.shape[1]] = U
        C = unitary_build(C.copy(), U.shape)


    sigma = np.zeros(min(A.shape))
    sigma[:r] = [np.sqrt(l) if l > 0 else 0 for l in lambdas ]
    Q = sort_by_permutation(sigma)
    sigma = np.diag(sigma)
    W_inv = C.H @ L_h @ P.T

    Q1 = np.eye(A.shape[0], A.shape[0])
    Q2 = np.eye(A.shape[1], A.shape[1])
    Sigma = np.eye(A.shape[0], A.shape[1])
    Sigma[:r, :r] = sigma[:r, :r]
    Q1[:Q.shape[0], :Q.shape[1]] = Q
    Q2[:Q.shape[0], :Q.shape[1]] = Q.T

    Sigma = (Q1 @ Sigma @ Q2)
    V = V @ Q1.T
    W_inv = Q2.T @ W_inv
    return np.array(V), np.array(Sigma), np.array(W_inv)


if __name__ == '__main__':
    n = 500
    A = np.random.rand(n, n)
    k = 500
    start = time.time()
    V, S, W = truncated_bk(A)
    print('BK in {} secs\n'.format(time.time() - start))

    A1 = V @ S @ W
    print(np.linalg.norm(A1 - A, 'fro'))

    f = np.linalg.norm(A, 'fro')
    start = time.time()
    s1, s2, s3 = np.linalg.svd(A, full_matrices=False)
    print('SVD in {} secs\n'.format(time.time() - start))
    s2 = np.diag(s2)
    es_bk = []
    es_svd = []
    r_bk = []
    r_svd = []

    for i in range(S.shape[0], 0, -1):
        A1 = V[:, :i] @ S[:i, :i] @ W[:i, :]
        A2 = s1[:, :i] @ s2[:i, :i] @ s3[:i, :]
        E = np.array(A - A1)
        E2 = np.array(A - A2)
        e1 = np.linalg.norm(E, 'fro')
        e2 = np.linalg.norm(E2, 'fro')

        es_bk.append(e1)
        es_svd.append(e2)
        r_bk.append(e1 / f * 100)
        r_svd.append(e2 / f * 100)


    p1 = mpatches.Patch(color=sns.xkcd_rgb["amber"], label='BK')
    p2 = mpatches.Patch(color=sns.xkcd_rgb["medium green"], label='SVD')
    plt.plot(list(range(k)), r_bk[:k], sns.xkcd_rgb["amber"], label='ap', linewidth=3)
    plt.plot(list(range(k)), r_svd[:k], sns.xkcd_rgb["medium green"], label='ap', linewidth=3)

    plt.ylabel('Ошибка аппроксимации, %')
    plt.xlabel('Усеченный ранг r')
    plt.legend(handles=[p1, p2])
    plt.show()


