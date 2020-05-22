import argparse
import sys
from cnn import CNN
from trainer import Trainer
from decomposer import Decomposer
from eval import predict, load_img, predict_loader, load_test_dataset
import torch
from torchsummary import summary

sys.path.append("/home/laneesra/PycharmProjects/Diplom/TensorTrain")
'''
num of images 25024

original:
2020-05-22 20:02:32,718 - Tensoring Nets - INFO - Mean time per image: 2.1943747562572753
2020-05-22 20:02:32,718 - Tensoring Nets - INFO - Accuracy of the network on the 7267 test images: 91 %
2020-05-22 20:02:32,718 - Tensoring Nets - INFO - predict_loader in 160.02049684524536 secs
accuracy: 91.15178202834733

cp layer=3 r=140:
Mean time per image: 1.8603774233863066
Accuracy of the network on the 7267 test images: 84 %
predict_loader in 138.02960801124573 secs
accuracy: 84.67042796202009

cp layer=3 rk=100:
Mean time per image: 1.8217532887687804
Accuracy of the network on the 7267 test images: 67 %
predict_loader in 135.23128724098206 secs
accuracy: 67.13912205862117

tt conv layer=3
Mean time per image: 2.239457038962334
Accuracy of the network on the 7267 test images: 78 %
2020-05-20 23:01:43,385 - Tensoring Nets - INFO - predict_loader in 170.85853385925293 secs
accuracy: 78.02394385578643

tt fc layer=4
Mean time per image: 7.382236154335015
Accuracy of the network on the 7267 test images: 50 %
2020-05-21 00:20:16,982 - Tensoring Nets - INFO - predict_loader in 539.6877450942993 secs
accuracy: 50.97013898445025

cp 3 conv trained
Mean time per image: 1.78910707269293
Accuracy of the network on the 7267 test images: 90 %
2020-05-21 11:47:01,652 - Tensoring Nets - INFO - predict_loader in 130.6662678718567 secs
accuracy: 90.17476262556764

tt 4 fc trained
2020-05-22 22:32:09,019 - Tensoring Nets - INFO - Mean time per image: 1.7881131811141968
2020-05-22 22:32:09,019 - Tensoring Nets - INFO - Accuracy of the network on the 25000 test images: 90 %
2020-05-22 22:32:09,019 - Tensoring Nets - INFO - predict_loader in 447.64107179641724 secs
accuracy: 90.26
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='alexnet', help='model architecture')
    parser.add_argument('--factorization', type=str, default='none', help='factorization method')
    parser.add_argument('--type', type=str, default='none', help='type of layer: fc or conv')
    parser.add_argument('-k', '--key', action='append', help='keys of layers')
    parser.add_argument("--decompose", dest="decompose", action="store_true", default=False)
    parser.add_argument("--fine_tune", dest="fine_tune", action="store_true", default=False)
    parser.add_argument("--train", dest="train", action="store_true", default=False)
    parser.add_argument("--eval", dest="eval", action="store_true")
    parser.add_argument("--train_path", type=str, default="train")
    parser.add_argument("--test_path", type=str, default="test")
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
        decomposer.replace_layer(keys=args.key, type=args.type)
        path_o = f'/home/laneesra/PycharmProjects/Diplom/CNNs/models/{args.model}_{args.factorization}_{args.key}_{args.type}.pth'
        torch.save(decomposer.model, path_o)
        print('===========saved============')

    elif args.eval:
        if args.factorization != 'none':
            path = f'/home/laneesra/PycharmProjects/Diplom/CNNs/models/{args.model}_{args.factorization}_{args.key}_{args.type}.pth'
        else:
            path = f'/home/laneesra/PycharmProjects/Diplom/CNNs/models/{args.model}.pth'
        path = "/home/laneesra/PycharmProjects/Diplom/CNNs/models/alexnet_tt.pth"
        #path = "/home/laneesra/PycharmProjects/Diplom/CNNs/models/alexnet.pth"
        #path = "/home/laneesra/PycharmProjects/Diplom/CNNs/models/alexnet_4_linear.pth"
        #path = "/home/laneesra/PycharmProjects/Diplom/CNNs/models/alexnet_cp_3_conv.pth"

        print('model:', path)
        model = torch.load(path)
        model.eval()

        device = 'cuda'
        if device == 'cpu':
            model.cpu()
            use_cuda = False
            device = torch.device("cpu")
            summary(model, (3, 224, 224), device='cpu')
        else:
            model.cuda()
            use_cuda = True
            device = torch.device("cuda")
            summary(model, (3, 224, 224), device='cuda')

        labels = ['cat', 'dog']
        #img1 = load_img('/home/laneesra/PycharmProjects/Diplom/CNNs/data/6zE76PpELRY.jpg')
        #img2 = load_img('/home/laneesra/PycharmProjects/Diplom/CNNs/data/photo_2020-05-04_13-55-56.jpg')
        #img3 = load_img('/home/laneesra/PycharmProjects/Diplom/CNNs/data/photo_2020-05-04_13-56-08.jpg')
        #for img in [img1, img2, img3]:
        #    pred = predict(model, labels, img, device, use_cuda)
        #    print(f'predict is {pred}')
        test_loader = load_test_dataset(args.test_path)
        accuracy = predict_loader(model, test_loader, device=device)
        print('accuracy:', accuracy)
