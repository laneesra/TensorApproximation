import numpy as np
import time
from scipy.linalg import inv
import math
from numpy.linalg import lstsq
from scipy.linalg import orth
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import matplotlib.patches as mpatches


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


def find_orth(M):
    rand_vec = np.random.rand(M.shape[0], 1)
    A = np.hstack((M, rand_vec))
    b = np.zeros(M.shape[1] + 1)
    b[-1] = 1.
    l = lstsq(A.T, b)[0]
    return l


if __name__ == '__main__':
    A = np.matrix(np.random.rand(500, 500))
    S = np.dot(A.H, A)
    #print('S', S)
    #print('shape', A.shape)
    start = time.time()
    D, L, P, pivot = bunch_kaufman(S)
    test = L @ D @ L.H
    ll = np.linalg.eig(test)[0]
    ll = [math.sqrt(ll[i]) for i in range(len(ll)) ]
    #print('eigen', ll)
    #print('singular', np.linalg.svd(A)[1])


    r = np.linalg.matrix_rank(A)
    #print(r)
    S_bk = P @ L @ D @ L.H @ P.T
    print('S_bk', np.linalg.norm(S_bk - S, 'fro'))
    #print('D', D)
    #print('L', L)
    #print('P', P)
    lambdas, U = np.linalg.eig(D)
    #Q = sort_by_permutation(lambdas)
    #print('lambdas', lambdas)
    l1 = [1. / math.sqrt(lambdas[i]) if i < r else 0 for i in range(len(lambdas)) ]
    #print('Q', Q)
    #print('res', np.dot(np.dot(Q, np.diag(lambdas)), Q.T))
    B = A @ P @ inv(L.H) @ U @ np.diag(l1)
    B = B[:r, :r]
    lambdas, U = lambdas[:r], U[:r]

    #print('B', B)
    #print('Er', B.H @ B)
    V = np.zeros((A.shape[0], A.shape[0]))
    V[:r, :r] = B
    V[r:, :] = np.random.rand(A.shape[0] - r, A.shape[0])
    #print(V.shape)
#    V[:, r] = [1, 0, 0]

    V = np.matrix(V)
    #print('V', V.H @ V)

    C = np.matrix(np.zeros((A.shape[1], A.shape[1])))
    C[:U.shape[0], :U.shape[1]] = U
    #C[r, :] = [0., 0., 0., 1.]
    #print('C', C.H @ C)

    sigma = np.zeros(min(A.shape))
    #print('sigma', lambdas)

    sigma[:r] = [math.sqrt(l) for l in lambdas]
    #print('sigma sqrt', sigma)

    Q = sort_by_permutation(sigma)
    sigma = np.diag(sigma)
    W_inv = C.H @ L.H @ P.T
    Q1 = np.eye(A.shape[0], A.shape[0])
    Q2 = np.eye(A.shape[1], A.shape[1])
    Sigma = np.eye(A.shape[0], A.shape[1])

    Sigma[:r, :r] = sigma[:r, :r]
    Q1[:Q.shape[0], :Q.shape[1]] = Q
    Q2[:Q.shape[0], :Q.shape[1]] = Q.T

#    A_ = V @ Q1.T @ (Q1 @ Sigma @ Q2) @ Q2.T @ W_inv
    Sigma = (Q1 @ Sigma @ Q2)
    #print('sigma sort', Sigma)

    #print('singular', np.linalg.svd(A)[1])
    #print(V.shape, Q1.T.shape)
    V = V @ Q1.T
    W_inv = Q2.T @ W_inv
    print('BK in {} secs\n'.format(time.time() - start))

    t = Tensor(A)
    f = t.frobenius_norm(A)
    start = time.time()
    s1, s2, s3 = np.linalg.svd(A)
    print('SVD in {} secs\n'.format(time.time() - start))
    s2 = np.diag(s2)
    es_bk = []
    es_svd = []
    r_bk = []
    r_svd = []
    for i in range(V.shape[1], 0, -1):
        A1 = V[:, :i] @ Sigma[:i, :i] @ W_inv[:i, :]
        A2 = s1[:, :i] @ s2[:i, :i] @ s3[:i, :]
        #print(A1, '\n', A2, '\n')
        E = np.array(A - A1)
        E2 = np.array(A - A2)
        e1 = np.sum(np.sum(E * E)) ** 0.5
        e2 = np.sum(np.sum(E2 * E2)) ** 0.5
        if i > 9:
            print(V.shape[1] - i, 'bk: %2f' % e1, 'svd: %2f' % e2)
            print(' rel: %2f' % (e1 / f), ' rel: %2f' % (e2 / f))
        else:
            print(V.shape[1] - i, 'bk: %2f' % e1, ' svd: %2f' % e2)
            print('  rel: %2f' % (e1 / f), ' rel: %2f' % (e2 / f))
        es_bk.append(e1)
        es_svd.append(e2)
        r_bk.append(e1 / f)
        r_svd.append(e2 / f)
        #print('%2f' % (e1 / f))
        #print('%2f' % (e2 / f))
        print()
    plt.plot(list(range(V.shape[1])), es_bk, sns.xkcd_rgb["amber"], label='ap', linewidth=3)
    for p in range(V.shape[1]):
        plt.plot([p], [es_bk[p]], 'o', color=sns.xkcd_rgb["dusty red"])
    plt.plot(list(range(V.shape[1])), es_svd, sns.xkcd_rgb["medium green"], label='ap', linewidth=3)
    for p in range(V.shape[1]):
        plt.plot([p], [es_svd[p]], 'o', color=sns.xkcd_rgb["windows blue"])
    plt.title('Error (Frobenius norm)')
    p1 = mpatches.Patch(color=sns.xkcd_rgb["amber"], label='Bunch-Kaufman')
    p2 = mpatches.Patch(color=sns.xkcd_rgb["medium green"], label='SVD')
    plt.legend(handles=[p1, p2])
    plt.show()

    plt.plot(list(range(V.shape[1])), r_bk, sns.xkcd_rgb["amber"], label='ap', linewidth=3)
    for p in range(V.shape[1]):
        plt.plot([p], [r_bk[p]], 'o', color=sns.xkcd_rgb["dusty red"])
    plt.plot(list(range(V.shape[1])), r_svd, sns.xkcd_rgb["medium green"], label='ap', linewidth=3)
    for p in range(V.shape[1]):
        plt.plot([p], [r_svd[p]], 'o', color=sns.xkcd_rgb["windows blue"])


    plt.title('Relative error)')
    plt.legend(handles=[p1, p2])
    plt.show()

    #print(A)