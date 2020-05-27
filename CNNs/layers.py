from torch import nn, randn, Tensor, matmul, from_numpy, add
import numpy as np
import math

from utils import timeit


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Parameter(randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1. / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    @timeit
    def forward(self, input):
        res = matmul(input, self.weight.t())
        if self.bias is not None:
            res = add(res, self.bias)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class TTLayer(nn.Module):
    def __init__(self, in_features, out_features, tt_ranks, bias=True):
        super(TTLayer, self).__init__()
        self.in_features = np.array(in_features)
        self.out_features = np.array(out_features)
        self.tt_ranks = np.array(tt_ranks)
        self.d = self.in_features.shape[0]
        self.shapes = [(self.in_features[k] * self.tt_ranks[k + 1], self.tt_ranks[k] * self.out_features[k])
                       for k in range(self.d - 1, -1, -1)]
        self.steps = [np.prod(shape) for shape in self.shapes]
        w = self.init_cores()
        self.weight = nn.Parameter(Tensor(w.shape))
        self.weight.data = from_numpy(w)
        self.weight.requires_grad = True

        if bias:
            shape = np.prod(out_features)
            self.bias = nn.Parameter(Tensor(shape))
            self.bias.requires_grad = True
            bound = 1.0 / np.sqrt(len(self.weight))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    @timeit
    def forward(self, input):
        res = input
        pos = 0
        for k in range(self.d - 1, -1, -1):
            core = self.weight[pos:pos + self.steps[k]].reshape(self.shapes[k])
            r = res.reshape((-1, self.shapes[k][0]))
            res = matmul(r, core)
            res = res.reshape((-1, self.out_features[k])).T
            pos += self.steps[k]

        res = res.reshape((-1, np.prod(self.out_features)))
        if self.bias is not None:
            res = add(res, self.bias)
        return res

    def extra_repr(self):
        return 'input_features={}, output_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def init_cores(self):
        cores = np.load('data/tt_fc_4_alexnet_cores.npy', allow_pickle=True)
        return cores
