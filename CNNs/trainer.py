import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms

sys.path.append("../Tensor")
from torch.utils.data.sampler import SubsetRandomSampler
from utils import logger
from torch.optim.lr_scheduler import StepLR


class Trainer:
    def __init__(self, args, model):
        self.verbose = args.verbose
        self.model = model
        self.path = f'models/{args.model}.pth'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            logger.debug('model {} loaded'.format(args.model))

        self.num_epochs = 100
        self.train_loader, self.test_loader = self.load_dataset(args.train_path)
        self.criterion = torch.nn.CrossEntropyLoss()
        learning_rate = 0.1
        self.optimizer = torch.optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
        #self.optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0001)

        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

        logger.info(self.optimizer)
        logger.info(self.criterion)
        self.model.cuda()

    def load_dataset(self, path, batch_size=64, num_workers=16, pin_memory=True, valid_size=.2):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_data = datasets.ImageFolder(path, transform)
        test_data = datasets.ImageFolder(path, transform)
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.seed(1234)

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
        test_accuracy = []
        logger.info('start train')
        total_steps = len(self.train_loader)
        logger.info(f'test_loader batches: {len(self.test_loader)}')
        logger.info(f'train_loader batches: {len(self.train_loader)}')

        for epoch in range(self.num_epochs):
            running_loss = 0
            test_loss = 0
            test_acc = 0
            steps = 0
            self.scheduler.step()
            logger.info(f'Epoch: {epoch} LR: {self.scheduler.get_lr()}')

            for inputs, labels in self.train_loader:
                steps += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                logger.info(f'step: {steps} / {total_steps} loss: {loss}')

                # validate
                if steps % print_every == 0:
                    print('steps', steps)
                    for p in self.model.parameters():
                        if p.grad is not None:
                            print(torch.sum(p.grad.data))

                    with torch.no_grad():
                        iterator = iter(self.test_loader)
                        inputs, labels = iterator.next()
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        batch_loss = self.criterion(outputs, labels)
                        test_loss += batch_loss.item()
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
                        test_acc += accuracy
                        logger.info(f'val_accuracy: {accuracy}  val_loss: {batch_loss}')

            train_losses.append(running_loss / len(self.train_loader))
            test_losses.append(test_loss / len(self.train_loader) * print_every)
            test_accuracy.append(test_acc / len(self.train_loader) * print_every)

            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}.. "
                        f"Train loss: {running_loss / total_steps}.. ")
            torch.save(self.model, self.path.replace('.pth', f'_epoch_{epoch}.pth'))

        self.train_losses = train_losses
        self.test_losses = test_losses
        self.accuracy = test_accuracy

        torch.save(self.model, self.path)

    def view(self):
        plt.plot(self.train_losses, label='Training loss')
        plt.plot(self.test_losses, label='Validation loss')
        plt.plot(self.accuracy, label='Validation accuracy')

        plt.legend(frameon=False)
        plt.savefig('training.png')
        plt.show()
