from cnn import CNN
from torchvision import transforms
from PIL import Image
import torch


def normalize_img(img):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])(img)


def load_img(src):
    img = Image.open(src)
    img_t = normalize_img(img)
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t


