import time

from torch import nn
from torch import Tensor
from torch import add
import torch
from torchvision import models
from torchsummary import summary
import numpy as np
import math
from utils import logger, timeit, matvec, device
import torch.nn.functional as F
from itertools import repeat
from torch._six import container_abcs


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    @timeit
    def forward(self, input):
        res = torch.matmul(input, self.weight.t())
        if self.bias is not None:
            res = add(res, self.bias)
        return res

    def extra_repr(self):
        return 'CustomLinear in_features={}, out_features={}, bias={}'.format(
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
        self.weight.data = torch.from_numpy(w)
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
            res = torch.matmul(r, core)
            res = res.reshape((-1, self.out_features[k])).T
            pos += self.steps[k]

        res = res.reshape((-1, np.prod(self.out_features)))
        if self.bias is not None:
            res = add(res, self.bias)
        return res

    def extra_repr(self):
        return 'TT-layer input_features={}, output_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def init_cores(self):
        np.random.seed(1234)
        cores_ar = np.zeros(np.sum(self.in_features * self.out_features * self.tt_ranks[1:] * self.tt_ranks[:-1]))
        cores = np.load('/home/laneesra/PycharmProjects/Diplom/CNNs/data/tt_fc_4_alexnet_cores.npy', allow_pickle=True)
        assert (len(cores_ar) == len(cores))
        cores_ar = cores
        return cores_ar


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
            w_p.data = torch.from_numpy(w)
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
        w = self.weight[0].data
        for i in range(1, len(self.weight)):
            w = np.tensordot(w, self.weight[i].data, [len(w.shape) - 1, 0])
        w = w.reshape((w.shape[3], w.shape[0], w.shape[1], w.shape[2]))
        w_t = Tensor(w.shape)
        w_t.data = torch.from_numpy(w)
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


class CNN(nn.Module):
    def __init__(self, model, num_classes=2):
        super(CNN, self).__init__()
        self.model_name = model

        if model == 'alexnet':
            self.features = models.alexnet(pretrained=True).features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.BatchNorm1d(4096),

                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096),

                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        elif model == 'alexnet_tt_fc':
            self.features = models.alexnet(pretrained=True).features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                TTLayer(in_features=[2, 2, 4, 4, 4, 4, 2, 2],
                        out_features=[2, 2, 4, 4, 4, 4, 2, 2],
                        tt_ranks=[1, 4, 16, 64, 512, 64, 16, 4, 1]),
                # nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

            # Freeze those weights
            # for p in self.features.parameters():
            #    p.requires_grad = False


        else:
            raise ("This architecture is not supported")

    def forward(self, input):
        f = self.features(input)
        if self.model_name == 'alexnet' or self.model_name == 'alexnet_tt':
            f = f.view(f.size(0), 256 * 6 * 6)  # reshape
        elif self.model_name == 'vgg16':
            f = f.view(f.size(0), -1)
        output = self.classifier(f)

        return output

    def get_summary(self):
        summary(self, (3, 224, 224), device='cpu')

        '''for name, param in self.model.named_parameters():
            print('name: ', name)
            print(type(param))
            print('param.shape: ', param.shape)
            print('param.requires_grad: ', param.requires_grad)
            print('=====')'''


