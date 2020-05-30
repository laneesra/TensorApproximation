import argparse
from cnn import CNN
from trainer import Trainer
from decomposer import Decomposer
from eval import predict, load_img, predict_loader, load_test_dataset
from utils import logger
import torch
from torchsummary import summary
from collections import defaultdict
'''
num of images 1024

original:
accuracy: 92.87109375

cp 3 conv trained:
accuracy: 90.17476262556764

tt 4 fc trained:
accuracy: 90.26
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='alexnet', help='model architecture')
    parser.add_argument('-f', '--factorization', type=str, default='none', help='factorization method: tt for cp')
    parser.add_argument('--type', type=str, default='none', help='type of layer: fc or conv')
    parser.add_argument('-k', '--key', action='append', help='keys of layers')
    parser.add_argument('-d', "--decompose", dest="decompose", action="store_true", default=False, help='mode for decomposition')
    parser.add_argument('-t', "--train", dest="train", action="store_true", default=False, help='mode for training')
    parser.add_argument('-e', "--eval", dest="eval", action="store_true", default=False, help='mode for evaluate')
    parser.add_argument("--train_path", type=str, default="data/train")
    parser.add_argument("--test_path", type=str, default="data/test")
    parser.add_argument("--device", type=str, default="cpu", help='gpu or cpu usage')
    parser.add_argument("--epochs", type=int, default=10, help='number of epochs for training')
    parser.add_argument('-v', "--verbose", dest="verbose", action="store_true", default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.train:
        model = CNN(args.model)
        trainer = Trainer(args, model)
        trainer.train()
        trainer.view()

    if args.decompose:
        path_i = f'models/{args.model}.pth'
        model = torch.load(path_i)
        decomposer = Decomposer(args, model)
        decomposer.run()
        if args.type == 'fc':
            params = defaultdict(dict)
            for k in args.key:
                params[k]['d'] = 6
                params[k]['tt_ranks'] = [1, 8, 8, 8, 8, 8, 1]

            decomposer.replace_layer(keys=args.key, type=args.type, params=params)
        else:
            decomposer.replace_layer(keys=args.key, type=args.type)

        path_o = f'models/{args.model}_{args.factorization}_{args.key}_{args.type}.pth'
        torch.save(decomposer.model, path_o)
        logger.info('===========saved============')
        logger.info(f'saved to {path_o}')

    elif args.eval:
        path = f'models/{args.model}.pth'

        logger.info(f'model: {path}')
        model = torch.load(path)
        model.eval()
        device_name = args.device
        if device_name == 'cpu':
            model.cpu()
            use_cuda = False

        else:
            model.cuda()
            use_cuda = True

        device = torch.device(device_name)
        summary(model, (3, 224, 224), device=device_name)

        labels = ['cat', 'dog']
        test_loader = load_test_dataset(args.test_path)
        accuracy = predict_loader(model, test_loader, device, args.verbose)
        print('accuracy:', accuracy)
