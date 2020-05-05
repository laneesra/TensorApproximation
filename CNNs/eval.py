from cnn import CNN
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch


def normalize_img(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
                 transforms.Scale(256),
                 transforms.RandomSizedCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 normalize,
             ])(img)


def load_img(src):
    img = Image.open(src)
    img_t = normalize_img(img)
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t


def predict(model, labels, batch):
    preds = model(batch)
    ps = torch.exp(preds)
    top_p, top_class = ps.topk(1, dim=1)
    return labels[top_class]\


