from __future__ import absolute_import
import torch.nn as nn
import math
import torch.nn.functional as F

__all__ = ["resnet", "one_resnet"]

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
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
        super(Bottleneck, self).__init__()
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
    ONE-ResNet with 3 branches.
    Each branch returns [logits, control_weight].
    Final ensemble is weighted sum of all branches.
    """
    def __init__(self, depth, num_classes=1000):
        super(ONE_ResNet, self).__init__()
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
        # Three parallel branches
        self.layer3_1 = self._make_layer(block, 64, n, stride=2); self.inplanes = fix_inplanes
        self.layer3_2 = self._make_layer(block, 64, n, stride=2); self.inplanes = fix_inplanes
        self.layer3_3 = self._make_layer(block, 64, n, stride=2)

        # Control vector for ensemble weighting
        self.control_v1 = nn.Linear(fix_inplanes, 3)
        self.bn_v1 = nn.BatchNorm1d(3)

        self.avgpool = nn.AvgPool2d(8)
        self.avgpool_c = nn.AvgPool2d(16)

        # Classifiers for each branch
        self.classifier3_1 = nn.Linear(64 * block.expansion, num_classes)
        self.classifier3_2 = nn.Linear(64 * block.expansion, num_classes)
        self.classifier3_3 = nn.Linear(64 * block.expansion, num_classes)

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
        # Shared stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)

        # Control vector
        x_c = self.avgpool_c(x).view(x.size(0), -1)
        x_c = F.softmax(F.relu(self.bn_v1(self.control_v1(x_c))), dim=1)

        # Branch outputs
        def branch_forward(layer, clf):
            out = layer(x)
            out = self.avgpool(out).view(out.size(0), -1)
            return clf(out)

        x_3_1 = branch_forward(self.layer3_1, self.classifier3_1)
        x_3_2 = branch_forward(self.layer3_2, self.classifier3_2)
        x_3_3 = branch_forward(self.layer3_3, self.classifier3_3)

        # Expand control weights to match logits
        x_c_1 = x_c[:,0].unsqueeze(1).expand_as(x_3_1)
        x_c_2 = x_c[:,1].unsqueeze(1).expand_as(x_3_2)
        x_c_3 = x_c[:,2].unsqueeze(1).expand_as(x_3_3)

        # Weighted ensemble
        x_m = x_c_1 * x_3_1 + x_c_2 * x_3_2 + x_c_3 * x_3_3

        return [x_3_1, x_c_1], [x_3_2, x_c_2], [x_3_3, x_c_3], x_m

def resnet(**kwargs):
    return ResNet(**kwargs)

def one_resnet(**kwargs):
    return ONE_ResNet(**kwargs)
