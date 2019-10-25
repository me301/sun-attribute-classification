from scripts.utils import SunDataset
from scripts.models import resnet, mobilenet
from scripts.train_model import train_model, test_model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import argparse


def main():
    parser = argparse.ArgumentParser(
            description='Train a model on sun attributes dataset')
    parser.add_argument('command', default='train',
                        help="'train' or 'eval'")
    parser.add_argument('--images', required=True,
                        metavar='path/to/dataset', help='Path to the dataset')
    parser.add_argument('--image_label_path', required=True,
                        metavar='path/to/images.mat',
                        help='Path to the images mat file')
    parser.add_argument('--output_labels', required=True,
                        metavar='path/to/labels.mat',
                        help='Path to the continous label mat file')
    parser.add_argument('-batch_size', default=32, type=int, metavar=32,
                        help='Batch size of the dataset')
    parser.add_argument('-val_size', default=0.15, type=float,
                        help='Validation dataset size')
    parser.add_argument('-model', default='resnet',
                        help='Use resnet or mobilenet')
    parser.add_argument('-num_layers', default=18, type=int,
                        help='Number of layers in resnet')
    parser.add_argument('-pretrained', default=False, type=bool,
                        help='Pretrained or not')
    parser.add_argument('-frozen_layers', default=0, type=int,
                        help='Number of layers to freeze in the model')
    parser.add_argument('-lr', default=0.1, type=float,
                        help='Learning rate of the model')
    parser.add_argument('-steps', default=1, type=int,
                        help='Number of epochs required to reduce learning\
                             rate')
    parser.add_argument('-gamma', default=0.01, type=float,
                        help='Value to reduce learning rate by')
    parser.add_argument('-epochs', default=5, type=int,
                        help='Number of epochs to train the model on')
    parser.add_argument('--path_to_model', default=None,
                        metavar='path/to/model',
                        help='Path to the pretrained model, if using one')

    args = parser.parse_args()

    im_label_path = args.image_label_path
    im_path = args.images
    label_path = args.output_labels

    my_dataset = SunDataset(im_label_path, im_path, label_path)

    # print(my_dataset.__getitem__(0)[1])
    if args.command == 'train':
        indices = np.random.permutation(len(my_dataset)).tolist()
        train_size = int(len(indices)*(1-args.val_size))
        test_size = int(len(indices)*(args.val_size))

        train_dataset = torch.utils.data.Subset(my_dataset,
                                                indices[:train_size])
        test_dataset = torch.utils.data.Subset(my_dataset,
                                               indices[-test_size:])

        test_dataset.dataset.transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])

        train_dataset.dataset.transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])

        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size,
                shuffle=True, num_workers=3)

        test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=3)

    elif args.command == 'test':
        test_loader = torch.utils.data.DataLoader(
                my_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=3)

    else:
        raise NameError

    if args.model == 'resnet':
        model = resnet(layers=args.num_layers, pretrained=args.pretrained)
        if args.path_to_model is not None:
            model.load_state_dict(torch.load(args.path_to_model))
    elif args.model == 'mobilenet':
        model = mobilenet(pretrained=args.pretrained)
        if args.path_to_model is not None:
            model.load_state_dict(torch.load(args.path_to_model))
    else:
        raise NameError
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    # print(model)

    count = 0
    for params in model.parameters():
        count += 1
        if count > args.frozen_layers:
            params.requires_grad = True
        else:
            params.requires_grad = False

    criterion = nn.BCELoss()

    if args.command == 'train':

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                               step_size=args.step,
                                               gamma=args.gamma)

        out_model = train_model(model, train_loader, test_loader, criterion,
                                optimizer_ft, device, exp_lr_scheduler,
                                train_dataset, test_dataset,
                                num_epochs=args.epochs)

        torch.save(out_model.state_dict(), r'saved_model\trained_model.pt')
    elif args.command == 'test':
        test_model(model, test_loader, criterion,
                   device, my_dataset)
    else:
        raise NameError


if __name__ == "__main__":
    main()
