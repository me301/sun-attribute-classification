from torchvision import models
import torch.nn as nn


def resnet(layers=18, pretrained=False):
    if layers == 18 and not pretrained:
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 102), nn.Sigmoid())
        return model
    elif layers == 18 and pretrained:
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 102), nn.Sigmoid())
        return model
    elif layers == 34 and not pretrained:
        model = models.resnet34()
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 102), nn.Sigmoid())
        return model
    elif layers == 34 and pretrained:
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 102), nn.Sigmoid())
        return model
    elif layers == 50 and not pretrained:
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 102), nn.Sigmoid())
        return model
    elif layers == 50 and pretrained:
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 102), nn.Sigmoid())
        return model
    else:
        raise NameError


def mobilenet(pretrained=False):
    if pretrained:
        model = models.mobilenet_v2(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Linear(num_ftrs, 102),
                                            nn.Sigmoid())
        return model
    else:
        model = models.mobilenet_v2(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Linear(num_ftrs, 102),
                                            nn.Sigmoid())
        return model
