import numpy as np
import bunch_kaufman
import scipy
import time
import sys
import torch
import matplotlib.patches as mpatches

sys.path.append("../CNNs")
from layers import TTLayer, LinearLayer

from utils import timeit, logger
from tensor import Tensor
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

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
    input = torch.Tensor((1, 4096))
    data = np.random.rand(1, 4096).astype(np.float32)
    input.data = torch.from_numpy(data)
    times_4 = []
    times_8 = []

    ds = [4, 6, 8, 10]
    rs1 = [[1, 8, 8, 8, 1],
          [1, 8, 8, 8, 8, 8, 1],
          [1, 8, 8, 8, 8, 8, 8, 8, 1],
          [1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1]]
    rs2 = [[1, 4, 4, 4, 1],
          [1, 4, 4, 4, 4, 4, 1],
          [1, 4, 4, 4, 4, 4, 4, 4, 1],
          [1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1]]
    rss = [rs2, rs1]
    times = [times_4, times_8]
    print('linear')
    layer = LinearLayer(4096, 4096)
    layer.weight.data = torch.from_numpy(weights)
    res = layer.forward(input)
    res = layer.forward(input)
    start = time.time()
    res = layer.forward(input)
    or_t = time.time() - start
    print('===========')
    rs3 = [or_t * 1000] * 4
    for j in range(2):
        rs = rss[j]
        for i in range(4):
            weight_tensor = Tensor(weights, from_matrix=True, d=ds[i])
            # [4, 32, 256, 1494, 4096, 512, 64, 16, 4]
            tt_ranks = rs[i]
            print('d=', ds[i])
            print('tt_ranks', tt_ranks)

            t_tensor = torch.Tensor((4096, 4096))
            t_tensor.data = torch.from_numpy(weights)

            Gs = weight_tensor.tt_with_ranks(tt_ranks)

            sum = 1
            for s in weight_tensor.T.shape:
                sum *= s
            logger.info(f'tt parameters: {len(Gs)}')

            np.save('../CNNs/data/tt_fc_4_alexnet_cores.npy', Gs)

            '''layer = LinearLayer(4096, 4096)
            layer.weight.data = torch.from_numpy(weights)
            start = time.time()
            res = layer.forward(input)
            logger.info(f'linear {time.time() - start}')'''

            tt_layer = TTLayer(in_features=weight_tensor.ns,
                               out_features=weight_tensor.ms,
                               tt_ranks=tt_ranks)

            #start = time.time()
            tt_layer.forward(input)
            #logger.info(f'tt_layer {time.time() - start}')

            #start = time.time()
            tt_layer.forward(input)
            #logger.info(f'tt_layer {time.time() - start}')


            start = time.time()
            tt_layer.forward(input)
            t = time.time() - start
            times[j].append(t * 1000)
            #logger.info(f'tt_layer {time.time() - start}')
            print('===========')

    plt.plot(ds, times_4, sns.xkcd_rgb["amber"], label='ap', linewidth=3)
    for p in range(len(ds)):
        plt.plot([ds[p]], [times_4[p]], 'o', color=sns.xkcd_rgb["amber"])

    plt.plot(ds, times_8, sns.xkcd_rgb["dusty red"], label='ap', linewidth=3)
    for p in range(len(ds)):
        plt.plot([ds[p]], [times_8[p]], 'o', color=sns.xkcd_rgb["dusty red"])

    plt.plot(ds, rs3, sns.xkcd_rgb["medium green"], label='ap', linewidth=3)
    for p in range(len(ds)):
        plt.plot([ds[p]], [rs3[p]], 'o', color=sns.xkcd_rgb["medium green"])

    p1 = mpatches.Patch(color=sns.xkcd_rgb["amber"], label='rk = 4')
    p2 = mpatches.Patch(color=sns.xkcd_rgb["dusty red"], label='rk = 8')
    p3 = mpatches.Patch(color=sns.xkcd_rgb["medium green"], label='Исходный слой')

    plt.xlabel('TT-ранг')
    plt.ylabel('Время работы, мс')

    plt.legend(handles=[p1, p2, p3])
    plt.show()

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

def test_cp():
    A = np.random.rand(10, 60, 7, 5).astype(np.float32)
    tensor = Tensor(A)
    times = []
    ers = []
    times_bk = []
    ers_bk = []

    rks = [10, 20, 30, 40, 60, 80, 100]
    for rk in rks:
        print(rk)
        start = time.time()
        decomposed_cp = tensor.cp_als(rk, ret_tensors=False, maxiter=200)
        times.append(time.time() - start)
        er = tensor.relative_error(decomposed_cp) * 100
        print('cp_als error', er)
        ers.append(er)

        start = time.time()
        decomposed_cp = tensor.cp_rand(rk, ret_tensors=False, maxiter=200)
        times_bk.append(time.time() - start)
        er = tensor.relative_error(decomposed_cp) * 100
        print('cp_rand error', er)
        ers_bk.append(er)

    plt.plot(rks, times_bk, sns.xkcd_rgb["amber"], label='ap', linewidth=3)
    for p in range(len(rks)):
        plt.plot([rks[p]], [times_bk[p]], 'o', color=sns.xkcd_rgb["amber"])
    plt.plot(rks, times, sns.xkcd_rgb["dusty red"], label='ap', linewidth=3)
    for p in range(len(rks)):
        plt.plot([rks[p]], [times[p]], 'o', color=sns.xkcd_rgb["dusty red"])

    p1 = mpatches.Patch(color=sns.xkcd_rgb["amber"], label='CPRAND')
    p2 = mpatches.Patch(color=sns.xkcd_rgb["dusty red"], label='CP-ALS')
    plt.xlabel('Ранг')
    plt.ylabel('Время работы, с')

    plt.legend(handles=[p1, p2])
    plt.show()

    plt.plot(rks, ers_bk, sns.xkcd_rgb["amber"], label='ap', linewidth=3)
    for p in range(len(rks)):
        plt.plot([rks[p]], [ers_bk[p]], 'o', color=sns.xkcd_rgb["amber"])
    plt.plot(rks, ers, sns.xkcd_rgb["dusty red"], label='ap', linewidth=3)
    for p in range(len(rks)):
        plt.plot([rks[p]], [ers[p]], 'o', color=sns.xkcd_rgb["dusty red"])

    plt.xlabel('Ранг')
    plt.ylabel('Ошибка аппроксимации, %')
    plt.legend(handles=[p1, p2])
    plt.show()

def test_tt_bk():
    A = np.random.rand(3, 10, 5).astype(np.float32)
    tensor = Tensor(A)
    decomposed_tt = tensor.tt_factorization(1, factor='bk')
    for d in decomposed_tt:
        print(d.shape)
    tt = decomposed_tt[0]
    for i in range(1, len(decomposed_tt)):
        tt = np.tensordot(tt, decomposed_tt[i], [len(tt.shape) - 1, 0])
    print('tt-svd error', tensor.frobenius_norm(A - tt.reshape(A.shape)))
    decomposed_tt = tensor.tt_factorization(1, factor='svd')
    for d in decomposed_tt:
        print(d.shape)
    tt = decomposed_tt[0]
    for i in range(1, len(decomposed_tt)):
        tt = np.tensordot(tt, decomposed_tt[i], [len(tt.shape) - 1, 0])
    print('tt-svd error', tensor.frobenius_norm(A - tt.reshape(A.shape)))

import pandas as pd
def view_results():
    names = ['ALEXNET', 'TT-4', 'CP-3', 'CP-6', 'CP-8', 'CP-3,6', 'CP-3,6,8', 'CP-3 TT-4',
             'CP-3,6 TT-4', 'CP-3,6,8 TT-4']
    params = [226.36, 162.36, 226.32, 224.61, 223.79, 224.57, 222, 162.32, 160.57, 158.01]
    time = [82, 67, 59, 63, 59, 53, 42, 54, 49, 39]
    d = {}
    filled_markers = ('o', 'v', '^', '<', 'X', '8', 's', 'p', 'P', 'h')
    d['size'] = params
    d['time'] = time
    d['Архитектура'] = names
    # sns scatter, with regression fit turned off
    #sns.set_style("whitegrid")
    #data = 'https://vincentarelbundock.github.io/Rdatasets/csv/carData/Burt.csv'
    #df = pd.read_csv(data, index_col=0)

    df = pd.DataFrame.from_dict(d)
    sns.scatterplot(x='size', y='time', style='Архитектура',
                    hue='Архитектура', data=df, markers=filled_markers, s=250)
    '''for p in range(len(names)):
        if names[p] == 'CP-3':
            plt.text(params[p] + 1, time[p] + 1, names[p], fontsize='14')
        else:
            plt.text(params[p] - 4, time[p] + 1, names[p], fontsize='14')

        plt.plot([params[p]], [time[p]], 'o', color=sns.xkcd_rgb["dusty red"])
    "'''
    plt.xlabel('Размер, МБ')
    plt.ylabel('Время выполнения, сек')
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "70"
    plt.show()

#test_tt_bk()
test_time()
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
