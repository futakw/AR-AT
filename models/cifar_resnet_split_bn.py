"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn_names=None):
        super(BasicBlock, self).__init__()
        if bn_names is None:
            raise ValueError("bn_names should be provided")
        
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.ModuleDict({
            bn_name: nn.BatchNorm2d(planes)
            for bn_name in bn_names
        })
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.ModuleDict({
            bn_name: nn.BatchNorm2d(planes)
            for bn_name in bn_names
        })

        self.shortcut = nn.Sequential()
        self.shortcut_bn = nn.ModuleDict({
                bn_name: nn.Sequential()
                for bn_name in bn_names
            })
        if stride != 1 or in_planes != self.expansion * planes:
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(
            #         in_planes,
            #         self.expansion * planes,
            #         kernel_size=1,
            #         stride=stride,
            #         bias=False,
            #     ),
            #     nn.BatchNorm2d(self.expansion * planes),
            # )
            self.shortcut = nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            self.shortcut_bn = nn.ModuleDict({
                bn_name: nn.BatchNorm2d(self.expansion * planes)
                for bn_name in bn_names
            })

    # def forward(self, x,  bn_name=None):
    def forward(self, x):
        # print("aaa")
        x, bn_name = x

        out = F.relu(self.bn1[bn_name](self.conv1(x)))
        out = self.bn2[bn_name](self.conv2(out))
        _x = self.shortcut(x)
        _x = self.shortcut_bn[bn_name](_x)
        out = out + _x
        out = F.relu(out)
        return out, bn_name


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bn_names=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.ModuleDict({
            bn_name: nn.BatchNorm2d(planes)
            for bn_name in bn_names
        })
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.ModuleDict({
            bn_name: nn.BatchNorm2d(planes)
            for bn_name in bn_names
        })
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        # self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.bn3 = nn.ModuleDict({
            bn_name: nn.BatchNorm2d(self.expansion * planes)
            for bn_name in bn_names
        })

        self.shortcut = nn.Sequential()
        self.shortcut_bn = nn.ModuleDict({
                bn_name: nn.Sequential()
                for bn_name in bn_names
            })
        if stride != 1 or in_planes != self.expansion * planes:
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(
            #         in_planes,
            #         self.expansion * planes,
            #         kernel_size=1,
            #         stride=stride,
            #         bias=False,
            #     ),
            #     nn.BatchNorm2d(self.expansion * planes),
            # )
            self.shortcut = nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            self.shortcut_bn = nn.ModuleDict({
                bn_name: nn.BatchNorm2d(self.expansion * planes)
                for bn_name in bn_names
            })

    # def forward(self, x, bn_name=None):
    def forward(self, x):
        x, bn_name = x

        out = F.relu(self.bn1[bn_name](self.conv1(x)))
        out = F.relu(self.bn2[bn_name](self.conv2(out)))
        out = self.bn3[bn_name](self.conv3(out))
        _x = self.shortcut(x)
        _x = self.shortcut_bn[bn_name](_x)
        out = out + _x
        out = F.relu(out)
        return out, bn_name


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        proj_dim=128,
        original=False,
        bn_names=None,
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.feature_dim = 512 * block.expansion
        self.proj_dim = proj_dim
        self.feat_dim = 512 * block.expansion

        if original:
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            # default: cifar10 version
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
        # self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = nn.ModuleDict({
            bn_name: nn.BatchNorm2d(64)
            for bn_name in bn_names
        })
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, bn_names=bn_names)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, bn_names=bn_names)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, bn_names=bn_names)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, bn_names=bn_names)
        self.linear = nn.Linear(self.feat_dim, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bn_name = "base"

    def set_bn_name(self, bn_name):
        self.bn_name = bn_name

    def reset_bn_name(self):
        self.bn_name = None

    def _make_layer(self, block, planes, num_blocks, stride, bn_names=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn_names=bn_names))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

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

    def forward(self, x, get_feat=False, bn_name=None):
        # print(bn_name, self.bn_name)
        if bn_name is None:
            bn_name = self.bn_name
        out_conv1 = F.relu(self.bn1[bn_name](self.conv1(x)))
        # for name, param in self.named_parameters():
        #     print(name, id(param), param.mean())
        #     break
        # print("- conv1(x):", self.conv1(x).mean())
        # print("- conv1.weight:", self.conv1.weight.mean())
        out_layer1, _ = self.layer1([out_conv1, bn_name])
        out_layer2, _ = self.layer2([out_layer1, bn_name])
        out_layer3, _ = self.layer3([out_layer2, bn_name])
        out_layer4, _ = self.layer4([out_layer3, bn_name])

        # out = F.avg_pool2d(out_layer4, 4)
        out = self.avgpool(out_layer4)
        feat = out.view(out.size(0), -1)
        out = self.linear(feat)

        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# test()

if __name__ == "__main__":
    model =  ResNet(BasicBlock, [2, 2, 2, 2], bn_names=["base", "adv"])
    print(model)

    x = torch.randn(1, 3, 32, 32)

    model.set_bn_name("adv")
    model.reset_bn_name()
    y = model(x)


    # # save
    # torch.save(
    #     model.state_dict(),
    #     "/home/fmg/waseda/MultiModalRobustness/ckpts/1018/cifar10/intial_weight/cifar_resnet18.pth",
    # )
