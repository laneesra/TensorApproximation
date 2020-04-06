import argparse
import sys
from cnn import CNN
from eval import load_img
sys.path.append("/home/laneesra/PycharmProjects/Diplom/TensorTrain")
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='alexnet', help='model architecture')
    parser.add_argument('--factorization', type=str, default='cp', help='factorization method')
    parser.add_argument("--decompose", dest="decompose", action="store_true")
    parser.add_argument("--fine_tune", dest="fine_tune", action="store_true")
    parser.add_argument("--eval", dest="eval", action="store_true")
    parser.add_argument("--train_path", type=str, default="train")
    parser.add_argument("--test_path", type=str, default="test")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(decompose=False)
    parser.set_defaults(fine_tune=False)
    parser.set_defaults(cp=False)
    parser.set_defaults(verbose=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = CNN(args)
    model.run()
    model.get_summary()
    if args.decompose:
        key = 0
        layers = model.get_conv_layers()
        decomposed = model.decompose_conv_layer(layers[key])
        print('===========decomposed============')
        model.model.features._modules[key] = decomposed
        torch.save(model.model, 'decomposed_model')
        print('===========saved============')

        # print(decomposed)
        # weigths = Tensor(layers[0].weight.data)
        # error = weigths.frobenius_norm(weigths.T - decomposed)
        # print(error)
    elif args.eval:
        img = load_img('/home/laneesra/PycharmProjects/Diplom/CNNs/data/6zE76PpELRY.jpg')
        pred = model.predict_decomposed(img)
        print(f'predict is {pred}')