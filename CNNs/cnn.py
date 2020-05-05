from torch import nn
from torchvision import models
from torchsummary import summary


class CNN(nn.Module):
    def __init__(self, model, num_classes=2):
        super(CNN, self).__init__()
        self.model_name = model

        if model == 'alexnet':
            self.features = models.alexnet(pretrained=True).features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        elif model == 'vgg16':
            self.features = models.vgg16(pretrained=True).features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        else:
            raise ("This architecture is not supported")
        # Freeze those weights
        #for p in self.features.parameters():
        #    p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6) # reshape
        elif self.model_name == 'vgg16':
            f = f.view(f.size(0), -1)
        y_pred = self.classifier(f)
        return y_pred

    def get_summary(self):
        summary(self, (3, 224, 224), device='cpu')

        '''for name, param in self.model.named_parameters():
            print('name: ', name)
            print(type(param))
            print('param.shape: ', param.shape)
            print('param.requires_grad: ', param.requires_grad)
            print('=====')'''
