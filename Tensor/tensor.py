import numpy as np
from numpy.linalg import matrix_rank, inv
from scipy.linalg import pinv
from tensorly import unfold
import sys
sys.path.append("../CNNs")
from utils import timeit, logger
from functools import reduce
from bunch_kaufman import truncated_bk

def prime_factors(n):
    factors = []
    while n % 2 == 0:
        factors.append(2)
        n = int(n / 2)

    for i in range(3, n + 1, 2):
        if n == 1:
            break
        while n % i == 0:
            factors.append(i)
            n /= i

    factors.append(1)
    return np.array(factors)


class Tensor:
    def __init__(self, ar, from_matrix=False, d=None):
        if from_matrix:
            ar = np.array(ar)
            assert(d is not None)
            if ar.shape == (4096, 4096):
                if d == 8:
                    self.ms = [2, 2, 4, 4, 4, 4, 2, 2]
                    self.ns = [4, 4, 2, 2, 2, 2, 4, 4]
                elif d == 6:
                    self.ms = [4, 4, 4, 4, 4, 4]
                    self.ns = [4, 4, 4, 4, 4, 4]
                elif d == 4:
                    self.ms = [8, 8, 8, 8]
                    self.ns = [8, 8, 8, 8]
                elif d == 10:
                    self.ms = [2, 2, 2, 2, 4, 4, 2, 2, 2, 2]
                    self.ns = [4, 2, 2, 2, 2, 2, 2, 2, 2, 4]
                else:
                    self.ms, self.ns = self.init_ranks(ar, d)
            else:
                self.ms, self.ns = self.init_ranks(ar, d)

            self.T = ar.reshape(np.array(self.ms) * np.array(self.ns))
            logger.info(f'new shape: {self.T.shape}')
        else:
            self.T = np.array(ar)

    @staticmethod
    def init_ranks(ar, d):
        m_fs, n_fs = prime_factors(ar.shape[0]), prime_factors(ar.shape[1])
        d = min([d, len(m_fs), len(n_fs)])
        step_m, step_n = len(m_fs) // d, len(n_fs) // d

        ms = [reduce(lambda a, b: a * b, m_fs[i * step_m:(i + 1) * step_m]) if i != d - 1 else
          reduce(lambda a, b: a * b, m_fs[i * step_m:]) for i in range(d)]
        ns = [reduce(lambda a, b: a * b, n_fs[i * step_n:(i + 1) * step_n]) if i != d - 1 else
          reduce(lambda a, b: a * b, n_fs[i * step_n:]) for i in range(d)]
        return ms, ns

    def frobenius_norm(self, A):
        return np.sum(np.sum(A * A)) ** 0.5

    def relative_error(self, T_):
        return self.frobenius_norm(self.T - T_) / self.frobenius_norm(self.T)

    @timeit
    def tt_factorization(self, eps, factor='svd'):
        d = len(self.T.shape)
        N = self.T.size
        r = np.zeros(d + 1, dtype=np.int)
        r[0] = 1
        n = self.T.shape
        G = []
        delta = eps / ((d - 1) ** 0.5) * self.frobenius_norm(self.T)
        C = self.T.copy()

        for k in range(1, d):
            C = C.reshape((r[k - 1] * n[k], int(float(N) / (r[k - 1] * n[k]))))
            U, S, Vt, rk = self.low_rank_approximation(C, delta, factor)
            r[k] = rk
            G.append(U.reshape(r[k - 1], n[k], r[k]))
            C = np.dot(S, Vt)
            N = float(N * r[k]) / (n[k] * r[k - 1])
        G.append(C)
        G[0] = G[0].reshape(G[0].shape[1:])
        return G

    @timeit
    def tt_with_ranks(self, tt_ranks):
        d = len(self.T.shape)
        n = np.array(self.T.shape)
        core = np.zeros(np.sum(tt_ranks[:-1] * n * tt_ranks[1:]), dtype='float32')
        C = self.T.copy()
        pos = 0
        for k in range(1, d):
            C = C.reshape((tt_ranks[k - 1] * n[k - 1], -1))
            U, S, Vt = np.linalg.svd(C, full_matrices=False)
            U = U[:, :tt_ranks[k]]
            S = np.diag(S[:tt_ranks[k]])
            Vt = Vt[:tt_ranks[k], :]

            step = tt_ranks[k - 1] * n[k - 1] * tt_ranks[k]
            core[pos:pos + step] = U.ravel()
            pos += step
            C = np.dot(S, Vt)

        core[pos:] = C.ravel()
        return core

    def low_rank_approximation(self, A, delta, factor='svd'):
        assert(factor in ['svd', 'aca', 'bk'])
        if factor == 'svd':
            U, S, Vt = np.linalg.svd(A, full_matrices=False)

            rk = len(S)
            er = 0
            delta *= delta
            for i in range(S.shape[0] - 1, 0, -1):
                er += S[i] ** 2
                if er > delta:
                    rk = i + 1
                    break
            return U[:, :rk], np.diag(S[:rk]), Vt[:rk, :], rk

        elif factor == 'aca':
            C, G, R = self.adaptive_cross_approximation(A, delta)
            rk = np.min(G.shape)
            return C[:, :rk], inv(G[:rk, :rk]), R[:rk, :], rk

        elif factor == 'bk':
            V, S, W = truncated_bk(A)
            rk = len(S)
            S = np.diag(S)
            er = 0
            delta *= delta / (W.shape[0] ** 2)
            for i in range(S.shape[0] - 1, 0, -1):
                er += S[i] ** 2
                if er > delta:
                    rk = i + 1
                    break
            return V[:, :rk], np.diag(S[:rk]), W[:rk, :], rk

    def adaptive_cross_approximation(self, A, eps):
        R = [A]
        I, J = [], []
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

    def init_factors(self, N, rank):
        As_init = [None] * N
        for i in range(N):
            As_init[i] = np.array(np.random.rand(self.T.shape[i], rank), dtype='float')
        return As_init

    def khatri_rao(self, As):
        N = As[0].shape[1]
        M = 1
        for i in range(len(As)):
            M *= As[i].shape[0]
        ns = np.arange(len(As))

        P = np.zeros((M, N), dtype=As[0].dtype)
        for n in range(N):
            ab = As[ns[0]][:, n]
            for j in range(1, len(ns)):
                ab = np.kron(ab, As[ns[j]][:, n])
            P[:, n] = ab
        return P

    @timeit
    def cp_als(self, rank, ret_tensors=False, maxiter=300):
        N = self.T.ndim
        As = self.init_factors(N, rank)
        iter = 0
        lambdas = np.ones(rank)

        while iter < maxiter:
            iter += 1
            for n in range(N):
                Z = self.khatri_rao([As[i] for i in range(self.T.ndim) if i != n])
                W = unfold(self.T, n).dot(Z)
                V = np.ones((rank, rank), dtype=np.float32)
                for i in (list(range(n)) + list(range(n + 1, N))):
                    V = V * np.dot(As[i].T, As[i])
                A_cur = np.dot(W, pinv(V))  # solve AV=W

                # normalize A update lambdas
                if iter == 0:
                    lambdas = np.sqrt((A_cur ** 2).sum(axis=0))
                else:
                    lambdas = A_cur.max(axis=0)
                    lambdas[lambdas < 1] = 1.
                As[n] = A_cur / lambdas

        if ret_tensors:
            return lambdas, As

        decomposed = np.zeros(self.T.shape)
        for i in range(rank):
            tmp = lambdas[i] * As[0][:, i]
            for j in range(1, len(As)):
                tmp = np.multiply.outer(tmp, As[j][:, i])
            decomposed += tmp
        logger.info(f'cp_als error: {self.relative_error(decomposed)}')
        return decomposed

    def sampled_khatri_rao(self, S, As, n):
        skr = None
        matrices = [As[i] for i in range(len(As)) if i != n]
        for i, (m, A) in enumerate(zip(S, matrices)):
            if skr is None:
                skr = A[m, :]
            else:
                skr = skr * A[m, :]
        return skr

    def get_samples(self, As, n, num_samples):
        samples = [np.random.randint(low=0, high=As[i].shape[0], size=num_samples, dtype=int) for i in range(len(As)) if i != n]
        return samples

    @timeit
    def cp_rand(self, rank, maxiter=300, ret_tensors=False):
        N = self.T.ndim
        As = self.init_factors(N, rank)
        iter = 0
        lambdas = np.ones(rank)
        num_samples = int(10 * rank * np.log(rank)) #max(self.T.shape) * 10
        while iter < maxiter:
            iter += 1

            for n in range(N):
                s_idxs = self.get_samples(As, n, num_samples=num_samples)
                Z = self.sampled_khatri_rao(s_idxs, As, n)
                s_idxs = [s.tolist() for s in s_idxs]
                s_idxs.insert(n, slice(None, None, None))
                s_idxs = tuple(s_idxs)
                X_s = self.T[s_idxs] if n else self.T[s_idxs].T
                X_s = np.dot(Z.T, X_s)
                A_cur = (np.dot(pinv(np.dot(Z.T, Z)), X_s)).T

                if iter == 0:
                    lambdas = np.sqrt((A_cur ** 2).sum(axis=0))
                else:
                    lambdas = A_cur.max(axis=0)
                    lambdas[lambdas < 1] = 1.
                As[n] = A_cur / lambdas

        if ret_tensors:
            return lambdas, As

        decomposed = np.zeros(self.T.shape)
        for i in range(rank):
            tmp = lambdas[i] * As[0][:, i]
            for j in range(1, len(As)):
                tmp = np.multiply.outer(tmp, As[j][:, i])
            decomposed += tmp
        logger.info(f'cp_rand error: {self.relative_error(decomposed)}')
        return decomposed