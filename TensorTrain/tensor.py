import numpy as np
from numpy.linalg import matrix_rank, inv


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
        if factor == 'Bunch-Kaufman':
            C, G, R = self.Bunch_Kaufman(A, delta)

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

    def Bunch_Kaufman(self, A, eps, alpha=(1 + 17 ** 0.5) / 8):
        S = np.dot(A, A.transpose())
        P = np.array([i for i in range(S.shape[0])])
        L = np.identity(S.shape[0])
        pivots = []

        def pivot_1(i):
            L[i + 1:, i] = S[i + 1:, i] / S[i, i]
            S[i + 1:, i + 1:] -= L[i + 1:, i]*S[i + 1:, i].T
            pivots.append(1)

        def pivot_2():
            pass

        def exchange_rows_cols(i, j):
             S[:, [i, j]] = S[:, [i, j]]
             S[[i, j], :] = S[[i, j], :]

        pass    # todo finish

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
