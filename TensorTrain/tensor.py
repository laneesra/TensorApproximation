import numpy as np
from numpy.linalg import matrix_rank, inv
import bunch_kaufman
from scipy.linalg import pinv
from tensorly import unfold


class Tensor:
    def __init__(self, ar):
        self.T = np.array(ar)

    def frobenius_norm(self, A):
        return np.sum(np.sum(A * A)) ** 0.5

    def tt_factorization(self, eps, factor='svd'):
        d = len(self.T.shape)
        N = self.T.size
        r = np.zeros(d + 1, dtype=np.int)
        r[0] = 1
        n = self.T.shape
        G = []
        delta = eps / (d - 1) ** 0.5 * self.frobenius_norm(self.T)
        C = self.T.copy()

        for k in range(1, d):
            C = C.reshape((r[k - 1] * n[k], int(N / (r[k - 1] * n[k]))))
            U, S, Vt, rk = self.low_rank_approximation(C, delta, factor)
            r[k] = rk
            G.append(U.reshape(r[k - 1], n[k], r[k]))
            if factor == 'svd':
                C = np.dot(np.diag(S), Vt)
            else:
                C = np.dot(S, Vt)
            N = (N * r[k]) / (n[k] * r[k - 1])
        G.append(C)
        tt = G[0]
        for i in range(1, len(G)):
            tt = np.tensordot(tt, G[i], [len(tt.shape) - 1, 0])
        return tt[0].reshape(self.T.shape)

    def low_rank_approximation(self, A, delta, factor='svd'):
        assert(factor in ['svd', 'aca'])

        if factor == 'svd':
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            rk = len(S)
            for i in range(len(S), 0, -1):
                A1 = np.dot(np.dot(U[:, :i], np.diag(S[:i])), Vt[:i, :])
                E = A - A1
                if self.frobenius_norm(E) > delta:
                    rk = i + 1
                    break
            return U[:, :rk], S[:rk], Vt[:rk, :], rk

        elif factor == 'aca':
            C, G, R = self.adaptive_cross_approximation(A, delta)
            rk = np.min(G.shape)
            return C[:, :rk], inv(G[:rk, :rk]), R[:rk, :], rk

    def adaptive_cross_approximation(self, A, eps):
            R = [A]
            I = []
            J = []
            k = 0

            while self.frobenius_norm(R[k]) > eps*self.frobenius_norm(A) or k == 0:
                k += 1
                ij = np.where(np.abs(R[k - 1]) == np.abs(R[k - 1]).max())
                i, j = ij[0][0], ij[1][0]
                delta = R[k - 1][i, j]
                if delta == 0:
                    break
                I.append(i)
                J.append(j)

                u = np.array([R[k - 1][:, j]])
                v = np.array([R[k - 1][i, :]]) / delta
                R.append(R[k - 1] - np.matmul(u.T, v))

            C = np.zeros((A.shape[0], len(J)))
            G = np.zeros((len(I), len(J)))
            R_ = np.zeros((len(I), A.shape[1]))

            for k in range(len(I)):
                C[:, k] = A[:, J[k]]
                R_[k, :] = A[I[k], :]
                for l in range(len(J)):
                    G[k][l] = A[I[k]][J[l]]
            return C, G, R_

    def init_factors(self, N, rank, mod):
        As_init = [None] * N
        if mod == 'random':
            for i in range(N):
                As_init[i] = np.array(np.random.rand(self.T.shape[i], rank), dtype=np.float32)

        return As_init

    def khatrirao(self, As):
        N = As[0].shape[1]
        M = 1
        for i in range(len(As)):
            M *= As[i].shape[0]
        order = np.arange(len(As))

        # preallocate
        P = np.zeros((M, N), dtype=As[0].dtype)
        for n in range(N):
            ab = As[order[0]][:, n]
            for j in range(1, len(order)):
                ab = np.kron(ab, As[order[j]][:, n])
            P[:, n] = ab
        return P

    def uttkrp(self, As, n):
        order = list(range(n)) + list(range(n + 1, self.T.ndim))
        Z = self.khatrirao(tuple(As[i] for i in order)) #reverse???
        return unfold(self.T, n).dot(Z)

    def cp_als(self, rank, init='random', maxiter=300):
        N = self.T.ndim
        As = self.init_factors(N, rank, init)
        iter = 0
        lambdas = np.ones(rank)

        while iter < maxiter:
            iter += 1

            for n in range(N):
                V = self.uttkrp(As, n)
                Z = np.ones((rank, rank), dtype=np.float32)
                for i in (list(range(n)) + list(range(n + 1, N))):
                    Z = Z * np.dot(As[i].T, As[i])
                A_cur = np.dot(V, pinv(Z))  # solve AZ=V

                # normalize A update lambdas
                if iter == 0:
                    lambdas = np.sqrt((A_cur ** 2).sum(axis=0))
                else:
                    lambdas = A_cur.max(axis=0)
                    lambdas[lambdas < 1] = 1
                As[n] = A_cur / lambdas
        return As, lambdas
