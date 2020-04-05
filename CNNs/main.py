import torch
from torchvision import models
from torchsummary import summary
import argparse
import numpy as np
import sys
sys.path.append("/home/laneesra/PycharmProjects/Diplom/TensorTrain")

from tensor import Tensor

class Trainer:
    def __init__(self, args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.model == 'alexnet':
            self.model = models.alexnet(pretrained=True).to(device)
        self.verbose = args.verbose
        self.factorization = args.factorization
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

            if isinstance(self.model.features._modules[key], torch.nn.modules.conv.Conv2d):
                conv_layer = self.model.features._modules[key]
                conv_layers.append(conv_layer)
                if self.verbose:
                    print(key)
                    print(conv_layer.weight.data)
                    print('=====')

        return conv_layers

    def decompose_conv_layer(self, layer):
        weights = layer.weight.data
        weight_tensor = Tensor(weights)
        if self.factorization == 'cp':
            rk = max(weight_tensor.T.shape) // 3
            decomposed = weight_tensor.cp_als(rk, init='random')

        elif self.factorization == 'tt':
            decomposed = weight_tensor.tt_factorization(0.01)

        error = weight_tensor.frobenius_norm(weight_tensor.T - decomposed)
        print(f'error is {error}')

        return decomposed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='alexnet', help='model architecture')
    parser.add_argument('--factorization', type=str, default='cp', help='factorization method')
    parser.add_argument("--decompose", dest="decompose", action="store_true")
    parser.add_argument("--fine_tune", dest="fine_tune", action="store_true")
    parser.add_argument("--train_path", type=str, default="train")
    parser.add_argument("--test_path", type=str, default="test")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(decompose=False)
    parser.set_defaults(fine_tune=False)
    parser.set_defaults(cp=False)
    parser.set_defaults(verbose=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    if args.decompose:
        layers = trainer.get_conv_layers()
        decomposed = trainer.decompose_conv_layer(layers[0])
        print('===========decomposed============')
        #print(decomposed)
        #weigths = Tensor(layers[0].weight.data)
        #error = weigths.frobenius_norm(weigths.T - decomposed)
        #print(error)