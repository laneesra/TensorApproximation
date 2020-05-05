import torch
from torchvision import models
import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
sys.path.append("/home/laneesra/PycharmProjects/Diplom/TensorTrain")
from tensor import Tensor
from torch.utils.data.sampler import SubsetRandomSampler


class Trainer:
    """
    all models are trained by ImageNet dataset
    todo:
     add logger
     add config file
    """

    def __init__(self, args, model):
        self.verbose = args.verbose
        self.model = model
        self.path = f'/home/laneesra/PycharmProjects/Diplom/CNNs/models/{args.model}.pth'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print('model {} loaded'.format(args.model))

        self.model.cuda()
        self.num_epochs = 10
        self.train_loader, self.test_loader = self.load_dataset(args.train_path)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.99)

    def load_dataset(self, path, batch_size=32, num_workers=1, pin_memory=True, valid_size = .2):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_data = datasets.ImageFolder(path,
                                 transforms.Compose([
                                     transforms.Scale(256),
                                     transforms.RandomSizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize,
                                 ]))
        test_data = datasets.ImageFolder(path,
                                 transforms.Compose([
                                     transforms.Scale(256),
                                     transforms.RandomSizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize,
                                 ]))
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)
        train_idx, test_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        train_loader = torch.utils.data.DataLoader(train_data,
                                                  sampler=train_sampler,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers,
                                                  pin_memory=pin_memory
                                                  )
        test_loader = torch.utils.data.DataLoader(test_data,
                                                 sampler=test_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 pin_memory=pin_memory
                                                 )
        return train_loader, test_loader

    def train(self):
        print_every = 10
        train_losses, test_losses = [], []
        print('start train')
        for epoch in range(self.num_epochs):

            running_loss = 0
            steps = 0
            for inputs, labels in self.train_loader:
                steps += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                print(f'step: {steps}  loss: {loss}')

                # validate
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in self.test_loader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        batch_loss = self.criterion(outputs, labels)
                        test_loss += batch_loss.item()
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    print(f'step: {steps}  accuracy: {accuracy}')

            train_losses.append(running_loss / len(self.train_loader))
            test_losses.append(test_loss / len(self.test_loader))

            print(f"Epoch {epoch + 1}/{self.num_epochs}.. "
                  f"Train loss: {running_loss / print_every}.. "
                  f"Test loss: {test_loss / len(self.test_loader)}.. "
                  f"Test accuracy: {accuracy / len(self.test_loader)}")
            torch.save(self.model, self.path.replace('.pth', f'_epoch_{epoch}.pth'))
        self.train_losses = train_losses
        self.test_losses = test_losses

        torch.save(self.model, self.path)

    def view(self):
        plt.plot(self.train_losses, label='Training loss')
        plt.plot(self.test_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()

    def fine_tune(self):
        pass


class Decomposer:
    def __init__(self, args, model):
        self.verbose = args.verbose
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.factorization = args.factorization
        if self.verbose:
            print('model {} loaded'.format(args.model))

    def replace_conv_layer(self, key=0):
        print('===========before============')
        print(self.model.features)
        print('===========before============')
        key = list(self.model.features._modules.keys())[key]
        if isinstance(self.model.features._modules[key], torch.nn.modules.conv.Conv2d):
            conv_layer = self.model.features._modules[key]
        else:
            raise ValueError('Key is not connect with convolution layer')

        decomposed = self.decompose_conv_layer(conv_layer)
        print('===========decomposed============')

        modules = []
        for i, k in enumerate(self.model.features._modules.keys()):
            if k != key:
                modules.append(self.model.features._modules[k])
            else:
                for layer in decomposed:
                    modules.append(layer)

        self.model.features = torch.nn.Sequential(*modules)
        print('===========replaced============')
        print('===========after============')
        print(self.model.features)
        print('===========after============')

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
                    print(f'{key} is conv layer')
                    print('=====')

        return conv_layers

    def run(self):
        self.model.eval()
        self.model.cpu()

    def decompose_conv_layer(self, layer):
        weights = layer.weight.data
        print('before', weights.shape)
        weight_tensor = Tensor(weights)

        if self.factorization == 'cp':
            rk = max(weight_tensor.T.shape) // 3
            lambdas, Us = weight_tensor.cp_rand(rk, init='random', ret_tensors=True)

            if self.verbose:
                decomposed = np.zeros(weight_tensor.T.shape)
                for i in range(rk):
                    tmp = lambdas[i] * Us[0][:, i]
                    for j in range(1, len(Us)):
                        tmp = np.multiply.outer(tmp, Us[j][:, i])
                    decomposed += tmp
                print('=========error============')
                print(weight_tensor.frobenius_norm(decomposed - weight_tensor.T))

            f_t, f_s, f_y, f_x = Us
            f_x = (np.array(f_x) * lambdas)
            f_y = np.array(f_y)
            f_s = np.array(f_s)
            f_t = np.array(f_t)

            t = weights.shape[0]     #num of filters
            d = layer.kernel_size[0]   #kernel size
            s = weights.shape[1]    #num of channels
            print(t, d, s)
            print('shapes', f_s.shape, f_y.shape, f_x.shape, f_t.shape)

            #s y x t
            k_s_layer = torch.nn.Conv2d(in_channels=s, out_channels=rk, kernel_size=1,
                                        padding=0, stride=1, dilation=layer.dilation, bias=False)

            k_y_layer = torch.nn.Conv2d(in_channels=rk, out_channels=rk, kernel_size=(d, 1),
                                        stride=1, padding=(layer.padding[0], 0),
                                        dilation=layer.dilation, groups=rk, bias=False)

            k_x_layer = torch.nn.Conv2d(in_channels=rk, out_channels=rk, kernel_size=(1, d),
                                        stride=layer.stride, padding=(0, layer.padding[0]),
                                        dilation=layer.dilation, groups=rk, bias=False)

            k_t_layer = torch.nn.Conv2d(in_channels=rk, out_channels=t, kernel_size=1,
                                        padding=0, stride=1, dilation=layer.dilation, bias=True)

            k_t_layer.bias.data = layer.bias.data

            f_s = np.reshape(f_s.T, [rk, s, 1, 1])
            f_y = np.reshape(f_y.T, [rk, 1, d, 1])
            f_x = np.reshape(f_x.T, [rk, 1, 1, d])
            f_t = np.reshape(f_t, [t, rk, 1, 1])

            print('shapes', f_s.shape, f_y.shape, f_x.shape, f_t.shape)

            print('DATA TYPE', k_s_layer.weight.shape, k_y_layer.weight.shape, k_x_layer.weight.shape, k_t_layer.weight.shape)

            k_s_layer.weight.data = torch.tensor(f_s).float()
            k_y_layer.weight.data = torch.tensor(f_y).float()
            k_x_layer.weight.data = torch.tensor(f_x).float()
            k_t_layer.weight.data = torch.tensor(f_t).float()
            print('DATA TYPE', k_s_layer.weight.shape, k_y_layer.weight.shape, k_x_layer.weight.shape, k_t_layer.weight.shape)

            new_layers = [k_s_layer, k_y_layer, k_x_layer, k_t_layer]

            return new_layers


        elif self.factorization == 'tt':
            decomposed = weight_tensor.tt_factorization(0.1)

        if self.verbose:
            error = weight_tensor.frobenius_norm(weight_tensor.T - decomposed)
            print(f'error is {error}')

        return decomposed
