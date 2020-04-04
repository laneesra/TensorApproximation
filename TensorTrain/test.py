import numpy as np
import bunch_kaufman
import scipy
import time

from tensor import Tensor
from svd import svd


A = np.zeros((3, 4, 2))
A[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
A[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]
print(A)
rk = np.min(A.shape)
print('rank', rk)
tensor = Tensor(A)

Us, lambdas = tensor.cp_als(rk, init='random')
print('Us', Us)
print('lambdas', lambdas)
T_ap = np.zeros(A.shape)
print(len(Us))
for i in range(rk):
    T1 = lambdas[i] * np.multiply.outer(Us[0][:, i], Us[1][:, i])
    T2 = np.multiply.outer(T1, Us[2][:, i])
    T_ap = T_ap + T2

print(T_ap)
print('cp_als error', A - T_ap)


'''start = time.time()
u, s, v = np.linalg.svd(A)
print('SVD singular values: {}\n'.format(s))
print('SVD in {} secs\n'.format(time.time() - start))
print('SVD singular values squares: {}\n'.format(s ** 2))

start = time.time()
D, L, P_ = bunch_kaufman.solve(S)
P = np.zeros(L.shape)
for i in range(len(P_)):
    P[i][P_[i]] = 1

print('BK in {} secs\n'.format(time.time() - start))

S_bk = np.matmul(np.matmul(np.matmul(np.matmul(P, L), D), L.transpose()), P.transpose())
print('Error: {}\n'.format(np.linalg.norm(S_bk - S)))
#print('D', D)
#print('L', L)

eig = np.linalg.eig(D)[0]
print('BK D eigenvalues: ', eig)

eig = np.linalg.eig(S)[0]
print('A^*A eigenvalues: ', eig)'''
