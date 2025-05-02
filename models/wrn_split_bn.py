import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, bn_names=None):
        super(BasicBlock, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1 = nn.ModuleDict({
            bn_name: nn.BatchNorm2d(in_planes)
            for bn_name in bn_names
        })
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        # self.bn2 = nn.BatchNorm2d(out_planes)
        self.bn2 = nn.ModuleDict({
            bn_name: nn.BatchNorm2d(out_planes)
            for bn_name in bn_names
        })
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        x, bn_name = x

        if not self.equalInOut:
            x = self.relu1(self.bn1[bn_name](x))
        else:
            out = self.relu1(self.bn1[bn_name](x))
        if self.equalInOut:
            out = self.relu2(self.bn2[bn_name](self.conv1(out)))
        else:
            out = self.relu2(self.bn2[bn_name](self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out), bn_name
        else:
            return torch.add(x, out), bn_name


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, bn_names=None):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate, bn_names=bn_names
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, bn_names=None):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                    bn_names=bn_names,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x)
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(
        self, depth, num_classes, widen_factor=1, dropRate=0.0, original=False, bn_names=None
    ):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        if original:
            self.conv1 = nn.Conv2d(
                3, nChannels[0], kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
            )
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, bn_names=bn_names)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, bn_names=bn_names)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, bn_names=bn_names)
        # global average pooling and classifier
        # self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.bn1 = nn.ModuleDict({
            bn_name: nn.BatchNorm2d(nChannels[3])
            for bn_name in bn_names
        })
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bn_name = None

    def set_bn_name(self, bn_name):
        self.bn_name = bn_name

    def reset_bn_name(self):
        self.bn_name = None

    def avg_pool_feature(self, o):
        """o: cpu feature map"""
        if len(o.shape) == 4:
            feat = self.avgpool(o).reshape(o.shape[0], -1)
        elif len(o.shape) == 3:
            feat = torch.mean(o, 1)
        elif len(o.shape) == 2:
            feat = o
        else:
            print(o.shape)
            raise ValueError
        return feat

    def forward(self, x, get_feat=False, layer=7, bn_name=None):
        if bn_name is None:
            bn_name = self.bn_name

        if layer <= 0:
            return x
        out_conv1 = self.conv1(x)
        if layer == 1:
            return out_conv1
        out_layer1, _ = self.block1([out_conv1, bn_name])
        if layer == 2:
            return out_layer1
        out_layer2, _ = self.block2([out_layer1, bn_name])
        if layer == 3:
            return out_layer2
        out_layer3, _ = self.block3([out_layer2, bn_name])
        if layer == 4:
            return out_layer3
        out_layer4 = self.relu(self.bn1[bn_name](out_layer3))
        if layer == 5:
            return out_layer4

        # out = F.avg_pool2d(out_layer4, 8)
        # feat = out.view(-1, self.nChannels)
        out = self.avgpool(out_layer4)
        feat = torch.flatten(out, 1)

        if layer == 6:
            return feat
        out = self.fc(feat)
        if get_feat:
            feat_dict = {
                "conv1": self.avg_pool_feature(out_conv1),
                "layer1": self.avg_pool_feature(out_layer1),
                "layer2": self.avg_pool_feature(out_layer2),
                "layer3": self.avg_pool_feature(out_layer3),
                "layer4": self.avg_pool_feature(out_layer4),
                "feature": feat,
            }
            return out, feat_dict
        else:
            return out
