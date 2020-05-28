from torch import nn, randn, Tensor, matmul, from_numpy, add
from torch.nn import functional as F
import numpy as np
import math

from utils import timeit, _pair


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

    #@timeit
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
        cores = np.load('../CNNs/data/tt_fc_4_alexnet_cores.npy', allow_pickle=True)
        return cores


class TTConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, weights=None,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(TTConvLayer, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._reversed_padding_repeated_twice = tuple(x for x in reversed(self.padding) for _ in range(2))

        if weights is None:
            weights = self.init_cores()
        self.weight = []
        for w in weights:
            w_p = nn.Parameter(Tensor(w.shape))
            w_p.requires_grad = True
            w_p.data = from_numpy(w)
            self.weight.append(w_p)

        if bias:
            self.bias = nn.Parameter(Tensor(out_channels))
            self.bias.requires_grad = True
            bound = 1.0 / np.sqrt(len(self.weight))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    @timeit
    def forward(self, input):
        #few parameters but too long
        w = self.weight[0].data
        for i in range(1, len(self.weight)):
            w = np.tensordot(w, self.weight[i].data, [len(w.shape) - 1, 0])
        w = w.reshape((w.shape[3], w.shape[0], w.shape[1], w.shape[2]))
        w_t = Tensor(w.shape)
        w_t.data = from_numpy(w)
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            Tensor(w_t), self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, Tensor(w_t), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        return '{}, {}, kernel_size={}, stride={}, padding={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding
        )

    def init_cores(self):
        cores = np.load('data/tt_conv_3_alexnet_cores.npy', allow_pickle=True)
        return cores
