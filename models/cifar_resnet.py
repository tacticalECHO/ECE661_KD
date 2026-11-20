"""CIFAR ResNets used by Experiment C.

The ResNet32x4 and ResNet8x4 definitions follow the model family used by the
official DKD implementation. The classifier size defaults to CIFAR-100.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, inputs):
        residual = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        if self.downsample is not None:
            residual = self.downsample(inputs)
        return F.relu(outputs + residual)


class CIFARResNet(nn.Module):
    """A 6n+2 basic-block ResNet for 32x32 CIFAR images."""

    def __init__(self, depth, channels, num_classes=100):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("Basic-block CIFAR ResNet depth must have form 6n+2")
        if len(channels) != 4:
            raise ValueError("channels must contain stem and three stage widths")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        blocks_per_stage = (depth - 2) // 6
        self.depth = depth
        self.stage_channels = tuple(channels)
        self.num_classes = num_classes
        self._current_channels = channels[0]

        self.conv1 = _conv3x3(3, channels[0])
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.layer1 = self._make_stage(channels[1], blocks_per_stage, stride=1)
        self.layer2 = self._make_stage(channels[2], blocks_per_stage, stride=2)
        self.layer3 = self._make_stage(channels[3], blocks_per_stage, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(channels[3], num_classes)

        self._initialize_weights()

    def _make_stage(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self._current_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self._current_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        blocks = [
            BasicBlock(
                self._current_channels,
                out_channels,
                stride,
                downsample=downsample,
            )
        ]
        self._current_channels = out_channels
        blocks.extend(
            BasicBlock(self._current_channels, out_channels)
            for _ in range(1, num_blocks)
        )
        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs, return_features=False):
        feature0 = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        feature1 = self.layer1(feature0)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        pooled = self.avgpool(feature3).flatten(1)
        logits = self.fc(pooled)

        if not return_features:
            return logits

        features = {
            "feats": [feature0, feature1, feature2, feature3],
            "pooled_feat": pooled,
        }
        return logits, features


def resnet8x4(num_classes=100):
    return CIFARResNet(
        depth=8,
        channels=(32, 64, 128, 256),
        num_classes=num_classes,
    )


def resnet32x4(num_classes=100):
    return CIFARResNet(
        depth=32,
        channels=(32, 64, 128, 256),
        num_classes=num_classes,
    )


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


if __name__ == "__main__":
    batch = torch.randn(2, 3, 32, 32)
    for builder in (resnet8x4, resnet32x4):
        model = builder()
        logits, features = model(batch, return_features=True)
        print(
            builder.__name__,
            "params=",
            count_parameters(model),
            "logits=",
            tuple(logits.shape),
            "features=",
            [tuple(feature.shape) for feature in features["feats"]],
        )
