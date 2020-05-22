from torchvision import transforms
from PIL import Image
import time
import torch
import torchvision.datasets as datasets
from utils import timeit, logger


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


def predict(model, labels, batch, device, use_cuda):
    batch = batch.to(device)
    with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
        preds = model(batch)
    print(prof)
    ps = torch.exp(preds)
    top_p, top_class = ps.topk(1, dim=1)
    return labels[top_class]


def load_test_dataset(path, batch_size=64, num_workers=1, pin_memory=True):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        test_data = datasets.ImageFolder(path,
                                 transforms.Compose([
                                     transforms.Scale(256),
                                     transforms.RandomSizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize,
                                 ]))
        test_loader = torch.utils.data.DataLoader(test_data,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 pin_memory=pin_memory
                                                 )
        return test_loader


@timeit
def predict_loader(model, loader, device, batch_size=64):
    print('num of batches', len(loader))
    print('num of images', len(loader) * batch_size)
    correct = 0
    total = 0
    all_time = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            start = time.time()
            outputs = model(inputs)
            t = time.time() - start
            logger.info(f'time: {t}')
            all_time += t
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total > 1000:
                break

    accuracy = 100 * float(correct) / total
    logger.info(f'Mean time per image: {100 * float(all_time) / total}' )

    logger.info(f'Accuracy of the network on the {total} test images: %d %%' % (accuracy))
    return accuracy

