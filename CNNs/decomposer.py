import numpy as np
import sys
sys.path.append("../Tensor")
from utils import timeit, logger, device, device_name
from tensor import Tensor
from torch import nn, tensor
from layers import TTLayer, LinearLayer, TTConvLayer


class Decomposer:
    """Decompose layers by tensor decompositon"""
    def __init__(self, args, model):
        self.verbose = args.verbose
        self.model = model
        self.device = device
        self.factorization = args.factorization
        if self.verbose:
            logger.info('model {} loaded'.format(args.model))

    @timeit
    def replace_layer(self, keys=['0'], type='conv', params={}):
        if self.verbose:
            logger.info('===========before============')
            logger.info(self.model.features)
            logger.info(self.model.classifier)
            logger.info('===========before============')

        logger.info('start decomposing')
        decomposed = {}
        if type == 'conv':
            mmodules = self.model.features._modules
            for key in keys:
                if isinstance(self.model.features._modules[key], nn.modules.conv.Conv2d):
                    conv_layer = self.model.features._modules[key]
                    if key in params:
                        decomposed[key] = self.decompose_conv_layer(conv_layer, params[key]['rk'])
                    else:
                        decomposed[key] = self.decompose_conv_layer(conv_layer)

        elif type == 'fc':
            mmodules = self.model.classifier._modules
            for key in keys:
                assert key in params, 'pass tt_ranks and d as params[key] dict'
                if isinstance(self.model.classifier._modules[key], nn.modules.linear.Linear):
                    fc_layer = self.model.classifier._modules[key]
                    decomposed[key] = self.decompose_fc_layer(fc_layer, key, params[key]['d'], params[key]['tt_ranks'])

        elif type == 'custom_fc':
            mmodules = self.model.classifier._modules
            for key in keys:
                if isinstance(self.model.classifier._modules[key], nn.modules.linear.Linear):
                    linear = self.model.classifier._modules[key]
                    new_linear = LinearLayer(linear.in_features, linear.out_features)
                    new_linear.weight = linear.weight
                    decomposed[key] = [new_linear]

        else:
            raise ValueError('Key is connect with incorrect layer')

        logger.info('end decomposing')

        modules = []
        for i, k in enumerate(mmodules.keys()):
            if k not in keys:
                modules.append(mmodules[k])
            else:
                modules += decomposed[k]

        if type == 'conv':
            self.model.features = nn.Sequential(*modules)
        else:
            self.model.classifier = nn.Sequential(*modules)

        logger.info('layers replaced')

        if self.verbose:
            logger.info('===========after============')
            logger.info(self.model.features)
            logger.info(self.model.classifier)
            logger.info('===========after============')

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
                    logger.info(f'{key} is conv layer')
        return conv_layers

    def run(self):
        self.model.eval()
        if device_name == 'cpu':
            self.model.cpu()
        else:
            self.model.cuda()

    @timeit
    def decompose_conv_layer(self, layer, rk=None):
        weights = layer.weight.data
        weight_tensor = Tensor(weights)
        if self.verbose:
            logger.info(f'before shape: {weights.shape}')

        if self.factorization == 'cp':
            if rk is None:
                rk = max(weight_tensor.T.shape) // 3
            lambdas, Us = weight_tensor.cp_rand(rk, ret_tensors=True)

            if self.verbose:
                decomposed = np.zeros(weight_tensor.T.shape)
                for i in range(rk):
                    tmp = lambdas[i] * Us[0][:, i]
                    for j in range(1, len(Us)):
                        tmp = np.multiply.outer(tmp, Us[j][:, i])
                    decomposed += tmp
                logger.info('============error============')
                logger.info(f'cp_rand error: {weight_tensor.frobenius_norm(weight_tensor.T - decomposed)}')
                logger.info(f'relative error: {weight_tensor.relative_error(decomposed)}')
                logger.info(f'original shape: {weight_tensor.T.shape}')
                for u in Us:
                    logger.info(f'factor shape: {u.shape}')

            f_t, f_s, f_y, f_x = Us
            f_x = f_x * lambdas

            t = weights.shape[0]     #num of filters
            d = layer.kernel_size[0]   #kernel size
            s = weights.shape[1]    #num of channels

            #s y x t
            k_s_layer = nn.Conv2d(in_channels=s, out_channels=rk, kernel_size=1,
                                        padding=0, stride=1, dilation=layer.dilation, bias=False)

            k_y_layer = nn.Conv2d(in_channels=rk, out_channels=rk, kernel_size=(d, 1),
                                        stride=1, padding=(layer.padding[0], 0),
                                        dilation=layer.dilation, groups=rk, bias=False)

            k_x_layer = nn.Conv2d(in_channels=rk, out_channels=rk, kernel_size=(1, d),
                                        stride=layer.stride, padding=(0, layer.padding[0]),
                                        dilation=layer.dilation, groups=rk, bias=False)

            k_t_layer = nn.Conv2d(in_channels=rk, out_channels=t, kernel_size=1,
                                        padding=0, stride=1, dilation=layer.dilation, bias=True)

            if layer.bias is not None:
                k_t_layer.bias.data = layer.bias.data

            if self.verbose:
                logger.info('after shape: ' + str((f_s.shape, f_y.shape, f_x.shape, f_t.shape)))
            f_s = np.reshape(f_s.T, [rk, s, 1, 1])
            f_y = np.reshape(f_y.T, [rk, 1, d, 1])
            f_x = np.reshape(f_x.T, [rk, 1, 1, d])
            f_t = np.reshape(f_t, [t, rk, 1, 1])

            k_s_layer.weight.data = tensor(f_s).float()
            k_y_layer.weight.data = tensor(f_y).float()
            k_x_layer.weight.data = tensor(f_x).float()
            k_t_layer.weight.data = tensor(f_t).float()

            new_layers = [k_s_layer, k_y_layer, k_x_layer, k_t_layer]
            return new_layers

        elif self.factorization == 'tt':
            Gs = weight_tensor.tt_factorization(0.01)
            tt = Gs[0]
            for i in range(1, len(Gs)):
                tt = np.tensordot(tt, Gs[i], [len(tt.shape) - 1, 0])

            if self.verbose:
                logger.info(f'original shape: {weight_tensor.T.shape}')
                sum = 1
                for s in weight_tensor.T.shape:
                    sum *= s
                logger.info(f'original parameters: {sum}')
                sum = 0
                for g in Gs:
                    sum_i = 1
                    for s in g.shape:
                        sum_i *= s
                    sum += sum_i
                    logger.info(g.shape)
                logger.info(f'tt parameters: {sum}')

                logger.info(f'tt shape {tt.reshape(weight_tensor.T.shape).shape}')
                logger.info(f'tt-svd error {weight_tensor.frobenius_norm(weight_tensor.T - tt.reshape(weight_tensor.T.shape))}')
                logger.info(f'tt-svd error {weight_tensor.relative_error(tt.reshape(weight_tensor.T.shape))}')

            tt_layer = TTConvLayer(in_channels=layer.in_channels, out_channels=layer.out_channels,
                                   kernel_size=layer.kernel_size, weights=Gs, stride=layer.stride,
                                   padding=layer.padding, dilation=layer.dilation, groups=layer.groups,
                                   bias=True, padding_mode=layer.padding_mode)
            return [tt_layer]

        else:
            raise ValueError('Not supported decomposition for this layer ')
    @timeit
    def decompose_fc_layer(self, layer, k, d, tt_ranks, ins=None, outs=None):
        weights = np.array(layer.weight.data)
        if self.verbose:
            logger.info(f'before {weights.shape}')
            logger.info(f'input shape {layer.in_features}')

        if self.factorization == 'tt':
            weight_tensor = Tensor(weights, from_matrix=True, d=d)
            #[4, 32, 256, 1494, 4096, 512, 64, 16, 4]
            if ins is None:
                ins = weight_tensor.ns
                outs = weight_tensor.ms

            Gs = weight_tensor.tt_with_ranks(tt_ranks)

            if self.verbose:
                sum = 1
                for s in weight_tensor.T.shape:
                    sum *= s
                logger.info(f'original parameters: {sum}')
                logger.info(f'tt parameters: {len(Gs)}')

            np.save(f'data/tt_fc_{k}_alexnet_cores.npy', Gs)
            tt_layer = TTLayer(in_features=ins,
                    out_features=outs,
                    tt_ranks=tt_ranks)
            return [tt_layer]

        elif self.factorization == 'svd':
            U, S, Vt = randomized_svd(weights,
                              n_components=2048,
                              random_state=None)
            logger.info(U.shape, S.shape, Vt.shape)
            US = U @ np.diag(S)
            w_ap = US @ Vt
            logger.info(f'original parameters {weights.shape[0] * weights.shape[1]}')
            logger.info(f'new parameters {US.shape[0] * US.shape[1] + Vt.shape[0] * Vt.shape[1]}')
            logger.info(f'error {np.linalg.norm(weights - w_ap) / np.linalg.norm(weights)}')

        else:
            raise ValueError('Not supported decomposition for this layer ')
