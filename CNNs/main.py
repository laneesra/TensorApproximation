import argparse
import sys
from cnn import CNN
from utils import Trainer, Decomposer
from eval import predict, load_img
import torch
from torchsummary import summary

sys.path.append("/home/laneesra/PycharmProjects/Diplom/TensorTrain")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='alexnet', help='model architecture')
    parser.add_argument('--factorization', type=str, default='none', help='factorization method')
    parser.add_argument("--decompose", dest="decompose", action="store_true", default=False)
    parser.add_argument("--fine_tune", dest="fine_tune", action="store_true", default=False)
    parser.add_argument("--train", dest="train", action="store_true", default=False)
    parser.add_argument("--eval", dest="eval", action="store_true")
    parser.add_argument("--train_path", type=str, default="train")
    #parser.add_argument("--test_path", type=str, default="test")
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.train:
        model = CNN(args.model)
        trainer = Trainer(args, model)
        trainer.train()
        trainer.view()

    if args.decompose:
        path_i = f'/home/laneesra/PycharmProjects/Diplom/CNNs/models/{args.model}.pth'
        model = torch.load(path_i)
        decomposer = Decomposer(args, model)
        decomposer.run()
        decomposer.replace_conv_layer(key=0)
        path_o = f'/home/laneesra/PycharmProjects/Diplom/CNNs/models/{args.model}_{args.factorization}.pth'
        torch.save(decomposer.model, path_o)
        print('===========saved============')

        # print(decomposed)
        # weigths = Tensor(layers[0].weight.data)
        # error = weigths.frobenius_norm(weigths.T - decomposed)
        # print(error)
    elif args.eval:
        if args.factorization != 'none':
            path = f'/home/laneesra/PycharmProjects/Diplom/CNNs/models/{args.model}_{args.factorization}.pth'
        else:
            path = f'/home/laneesra/PycharmProjects/Diplom/CNNs/models/{args.model}.pth'

        model = torch.load(path)
        model.eval()
        model.cpu()
        summary(model, (3, 224, 224), device='cpu')
        print('TYPE', type(model.features))

        labels = ['cat', 'dog']
        img1 = load_img('/home/laneesra/PycharmProjects/Diplom/CNNs/data/6zE76PpELRY.jpg')
        img2 = load_img('/home/laneesra/PycharmProjects/Diplom/CNNs/data/photo_2020-05-04_13-55-56.jpg')
        img3 = load_img('/home/laneesra/PycharmProjects/Diplom/CNNs/data/photo_2020-05-04_13-56-08.jpg')
        for img in [img1, img2, img3]:
            pred = predict(model, labels, img)
            print(f'predict is {pred}')
        #for i in range(1, 30):

        #    img = load_img(f'/home/laneesra/PycharmProjects/Diplom/CNNs/data/test/{i}.jpg')
        #    pred = predict(model, labels, img)

        #   print(f'{i} predict is {pred}')