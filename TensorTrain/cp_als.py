import logging
from scipy.io.matlab import loadmat
from sktensor import dtensor, cp_als
from tensorly.decomposition import randomised_parafac as cp_rand
import numpy as np
import time
# Set logging to DEBUG to see CP-ALS information
logging.basicConfig(level=logging.DEBUG)
Tp = np.zeros((3, 4, 2))
Tp[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
Tp[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]
print(Tp)

T = dtensor(Tp)

# Decompose tensor using CP-ALS
start = time.time()
P, fit, itr = cp_rand(T, 4, n_samples=max(T.shape) + 1, init='random')

T_ap = np.zeros((3, 4, 2))
for i in range(4):
    print('shapes', P.U[0][:, i].shape, P.U[1][:, i].shape, P.U[2][:, i].shape)
    T1 = P.lmbda[i] * np.multiply.outer(P.U[0][:, i], P.U[1][:, i])
    T2 = np.multiply.outer(T1, P.U[2][:, i])
    print('1', T1.shape)
    print('2', T2.shape)
    T_ap = T_ap + T2

print(f'{time.time() - start} for {itr} iterations')
print(T_ap, T_ap.shape)
print(Tp - T_ap)