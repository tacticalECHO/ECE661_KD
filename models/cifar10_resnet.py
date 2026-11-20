import torch.nn as nn
from torchvision.models import resnet34, resnet50

def build_model(name):
    constructor = {"resnet34": resnet34, "resnet50": resnet50}[name]
    model = constructor(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    return model
