import torch
from torchvision import models
from torchsummary import summary
import argparse
import numpy as np
import sys
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import torch.tensor as torch_tensor
sys.path.append("/home/laneesra/PycharmProjects/Diplom/TensorTrain")
from tensor import Tensor
from tensorly.decomposition import parafac, partial_tucker


class CNN:
    """
    all models are trained by ImageNet dataset
    todo add logger
    """

    def __init__(self, args):
        device = torch.device('cuda' if cuda.is_available() else 'cpu')
        if args.model == 'alexnet':
            self.model = models.alexnet(pretrained=True).to(device)
        elif args.model == 'googlenet':
            self.model = models.googlenet(pretrained=True).to(device)

        self.verbose = args.verbose
        self.factorization = args.factorization

        with open('/home/laneesra/PycharmProjects/Diplom/CNNs/data/imagenet_classes.txt') as f:
            self.labels = [line.strip() for line in f.readlines()]
        '''if self.verbose:
            print('========ImageNet classes========')
            print(self.labels)
            print('================================')'''

    def run(self):
        self.model.eval()
        self.model.cpu()

    def get_summary(self):
        for name, param in self.model.named_parameters():
            print('name: ', name)
            print(type(param))
            print('param.shape: ', param.shape)
            print('param.requires_grad: ', param.requires_grad)
            print('=====')

    def get_conv_layers(self):
        N = len(self.model.features._modules.keys())
        conv_layers = []

        for i, key in enumerate(self.model.features._modules.keys()):
            if i >= N - 2:
                break

            if isinstance(self.model.features._modules[key], nn.modules.conv.Conv2d):
                conv_layer = self.model.features._modules[key]
                conv_layers.append(conv_layer)
                if self.verbose:
                    print(f'{key} is conv layer')
                    print('=====')

        return conv_layers

    def decompose_conv_layer(self, layer):
        weights = layer.weight.data
        print('before', weights.shape)
        weight_tensor = Tensor(weights)
        print()
        if self.factorization == 'cp':
            rk = max(weight_tensor.T.shape) // 3
            lambdas, Us = weight_tensor.cp_als(rk, init='random', ret_tensors=True)
            last, first, vertical, horizontal = Us

            f_x = (np.array(horizontal) * lambdas).T
            f_y = np.array(vertical).T
            f_c = np.array(first).T
            f_n = np.array(last)

            n = weights.shape[0]     #num of filters
            d = layer.kernel_size[0]   #kernel size
            c = weights.shape[1]
            print(f_y.shape, [rk, 1, d, 1])
            print(f_x.shape, [rk, 1, 1, d])
            print(f_c.shape, [rk, c, 1, 1])
            print(f_n.shape, [n, rk, 1, 1])

            f_y = np.reshape(f_y, [rk, 1, d, 1])
            f_x = np.reshape(f_x, [rk, 1, 1, d])
            f_c = np.reshape(f_c, [rk, c, 1, 1])
            f_n = np.reshape(f_n, [n, rk, 1, 1])

            #c y x n
            pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=f_c.shape[0], \
                                                     out_channels=f_c.shape[1], kernel_size=1, stride=1, padding=0,
                                                     dilation=layer.dilation, bias=False)

            depthwise_vertical_layer = torch.nn.Conv2d(in_channels=f_y.shape[0],
                                                       out_channels=f_y.shape[1],
                                                       kernel_size=(d, 1),
                                                       stride=1, padding=(layer.padding[0], 0), dilation=layer.dilation,
                                                       groups=f_y.shape[1], bias=False)

            depthwise_horizontal_layer = \
                torch.nn.Conv2d(in_channels=f_x.shape[0], \
                                out_channels=f_x.shape[1],
                                kernel_size=(1, d), stride=layer.stride,
                                padding=(0, layer.padding[0]),
                                dilation=layer.dilation, groups=f_x.shape[1], bias=False)

            pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=f_n.shape[0], \
                                                     out_channels=f_n.shape[1], kernel_size=1, stride=1,
                                                     padding=0, dilation=layer.dilation, bias=True)

            pointwise_r_to_t_layer.bias.data = layer.bias.data

            depthwise_horizontal_layer.weight.data = torch.tensor(f_c)
            depthwise_vertical_layer.weight.data = torch.tensor(f_y)
            pointwise_s_to_r_layer.weight.data = torch.tensor(f_x)
            pointwise_r_to_t_layer.weight.data = torch.tensor(f_n)

            new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer, \
                          depthwise_horizontal_layer, pointwise_r_to_t_layer]

            return nn.Sequential(*new_layers)


        elif self.factorization == 'tt':
            decomposed = weight_tensor.tt_factorization(0.01)

        if self.verbose:
            error = weight_tensor.frobenius_norm(weight_tensor.T - decomposed)
            print(f'error is {error}')

        return decomposed

    def fine_tune(self):
        base_model = torch.load("decomposed_model")
        model = torch.nn.DataParallel(base_model)

        for param in model.parameters():
            param.requires_grad = True

        print(model)

    def predict(self, batch):
        preds = self.model(batch)
        _, index = torch.max(preds, 1)
        percentage = nn.functional.softmax(preds, dim=1)[0] * 100
        return (self.labels[index[0]], percentage[index[0]].item())

    def predict_decomposed(self, batch):
        self.model = torch.load("decomposed_model")
       # summary(self.model, (3, 224, 224))
        self.run()
        preds = self.model(batch)
        _, index = torch.max(preds, 1)
        percentage = nn.functional.softmax(preds, dim=1)[0] * 100
        return (self.labels[index[0]], percentage[index[0]].item())
