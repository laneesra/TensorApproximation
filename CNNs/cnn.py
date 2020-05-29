from torch import nn
from torchvision import models
from torchsummary import summary
from utils import device_name


class CNN(nn.Module):
    """Convolution neural nets class for classification
    model_name: name of model architecture
    features: sequence of conv and other layers to get feature map
    classifier: sequence of fc and other layers to classify
    """

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

        else:
            raise ('This architecture is not supported')

    def forward(self, input):
        """Forward pass: compute predicted y"""
        f = self.features(input)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)  # reshape
        elif self.model_name == 'vgg16':
            f = f.view(f.size(0), -1)
        output = self.classifier(f)

        return output

    def get_summary(self):
        """Prints model info (layers and params size)"""
        summary(self, (3, 224, 224), device=device_name)



