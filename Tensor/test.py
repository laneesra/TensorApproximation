import numpy as np
import bunch_kaufman
import scipy
import time
import sys
import torch

sys.path.append("/home/laneesra/PycharmProjects/Diplom/CNNs")
from cnn import TTLayer, LinearLayer

from utils import timeit, logger
from tensor import Tensor
from svd import svd


def test_full_tt_svd():
    np.random.seed(1234)
    W = np.random.rand(4096, 4096).astype(np.float32)
    tensor = Tensor(W, from_matrix=True, d=6)
    Gs = tensor.tt_factorization(0.1)

    sum = 1
    for s in tensor.T.shape:
        sum *= s
    print('original parameters', sum)

    sum = 0
    for g in Gs:
        sum_i = 1
        for s in g.shape:
            sum_i *= s
        sum += sum_i
        print(g.shape)
    print('tt parameters', sum)

    tt = Gs[0]
    print(len(Gs))
    for i in range(1, len(Gs)):
        if i == 1 or i == 2:
            r = max(Gs[i].shape) // 3
            t = Tensor(Gs[i])
            cp_rand = t.cp_rand(r, init='random', ret_tensors=False)
            tt = np.tensordot(tt, cp_rand, [len(tt.shape) - 1, 0])

        else:
            tt = np.tensordot(tt, Gs[i], [len(tt.shape) - 1, 0])

    print('tt-svd error', tensor.frobenius_norm(tensor.T - tt))
    print('tt-svd error', tensor.relative_error(tt))


def test_time():
    weights = np.random.rand(4096, 4096).astype(np.float32)
    weight_tensor = Tensor(weights, from_matrix=True, d=8)
    # [4, 32, 256, 1494, 4096, 512, 64, 16, 4]
    tt_ranks = [1, 4, 4, 4, 4, 4, 4, 4, 1]
    input = torch.Tensor((1, 4096))
    data = np.random.rand(1, 4096).astype(np.float32)
    input.data = torch.from_numpy(data)
    t_tensor = torch.Tensor((4096, 4096))
    t_tensor.data = torch.from_numpy(weights)



    Gs = weight_tensor.tt_with_ranks(tt_ranks)
    #tt = Gs[0]
    #for i in range(1, len(Gs)):
    #    tt = np.dot(tt, Gs[i])

#    logger.info(f'tt-svd error {weight_tensor.frobenius_norm(weight_tensor.T - tt)}')
#    logger.info(f'tt-svd error {weight_tensor.relative_error(tt)}')

    sum = 1
    for s in weight_tensor.T.shape:
        sum *= s
    logger.info(f'original parameters: {sum}')
    logger.info(f'tt parameters: {len(Gs)}')

    np.save('/home/laneesra/PycharmProjects/Diplom/CNNs/data/tt_fc_4_alexnet_cores.npy', Gs)

    #layer = LinearLayer(4096, 4096)
    #layer.weight.data = torch.from_numpy(weights)
    #start = time.time()
    #res = layer.forward(input)
    #logger.info(f'linear {time.time() - start}')

    tt_layer = TTLayer(in_features=[2, 2, 4, 4, 4, 4, 2, 2],
                       out_features=[4, 4, 2, 2, 2, 2, 4, 4],
                       tt_ranks=tt_ranks)

    start = time.time()
    tt_layer.forward(input)
    logger.info(f'tt_layer {time.time() - start}')


def test_tt():
    A = np.random.rand(3, 5, 100, 200).astype(np.float32)
    tensor = Tensor(A)
    decomposed_tt = tensor.tt_factorization(0.01)
    for d in decomposed_tt:
        print(d.shape)
    tt = decomposed_tt[0]
    for i in range(1, len(decomposed_tt)):
        tt = np.tensordot(tt, decomposed_tt[i], [len(tt.shape) - 1, 0])
        print(tt.shape)
    print('tt-svd error', tensor.frobenius_norm(A - tt.reshape(A.shape)))
test_full_tt_svd()
'''A = np.zeros((3, 4, 2))
A[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
A[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]
print(A)
rk = 4
print('rank', rk)
tensor = Tensor(A)
decomposed_cp = tensor.cp_als(rk, init='random', ret_tensors=False)
print('cp_als error', tensor.frobenius_norm(A - decomposed_cp))

print('============')
decomposed_tt = tensor.tt_factorization(0.01)
print('tt-svd error', tensor.frobenius_norm(A - decomposed_tt))

print('============')
cp_rand = tensor.cp_rand(rk, init='random', ret_tensors=False)
#print(cp_rand)
print('cp_rand error', tensor.frobenius_norm(A - cp_rand))'''

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
