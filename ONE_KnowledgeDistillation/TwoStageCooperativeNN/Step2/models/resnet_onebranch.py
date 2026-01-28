"""
resnet_onebranch.py

Simplified ONE_ResNet model for CIFAR datasets with a single branch.
This version removes extra branches (layer3_2 ... layer3_5) and classifiers,
keeping only layer3_1 and classifier3_1.

Used as baseline for cooperative learning experiments.
"""

import torch.nn as nn
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ONE_ResNet(nn.Module):
    """
    ONE ResNet with a single branch (layer3_1).
    """
    def __init__(self, depth, num_classes=100):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth should be 6n+2"
        n = (depth - 2) // 6
        block = Bottleneck if depth >= 44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)

        fix_inplanes = self.inplanes
        self.layer3_1 = self._make_layer(block, 64, n, stride=2)

        self.control_v1 = nn.Linear(fix_inplanes, 1)
        self.bn_v1 = nn.BatchNorm1d(1)

        self.avgpool = nn.AvgPool2d(8)
        self.avgpool_c = nn.AvgPool2d(16)

        self.classfier3_1 = nn.Linear(64 * block.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n_w = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n_w))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)

        # Control branch
        x_c = self.avgpool_c(x).view(x.size(0), -1)
        x_c = F.relu(self.bn_v1(self.control_v1(x_c)))
        x_c = F.softmax(x_c, dim=1)

        # Branch 1
        x_3_1 = self.layer3_1(x)
        x_3_1 = self.avgpool(x_3_1).view(x_3_1.size(0), -1)
        x_3_1 = self.classfier3_1(x_3_1)

        # Weighted ensemble output
        x_c_1 = x_c[:, 0].repeat(x_3_1.size()[1], 1).transpose(0, 1)
        x_m = x_c_1 * x_3_1

        return x_3_1, x_m


def resnet(**kwargs):
    """Constructs a standard ResNet model."""
    return ResNet(**kwargs)


def one_resnet(**kwargs):
    """Constructs a ONE_ResNet model (single branch)."""
    return ONE_ResNet(**kwargs)
