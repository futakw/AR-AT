from .cifar_resnet import BasicBlock, Bottleneck, ResNet
from .feature_extractor import FeatureExtractor, FeatureExtractorSplitBN
from .wrn import WideResNet

import sys

# sys.path.append('DM-Improves-AT')
# from core.models.wideresnetwithswish import wideresnetwithswish


def get_model(args, name, num_classes):
    if name == "resnet18":
        net = ResNet(
            BasicBlock,
            [2, 2, 2, 2],
            num_classes=num_classes,
            original=True,
        )
    elif name == "wrn_34_10":
        net = WideResNet(
            34,
            num_classes,
            10,
            dropRate=0.0,
            original=True,
        )
    elif name == "cifar_resnet18":
        net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif name == "cifar_wrn_34_10":
        net = WideResNet(
            34,
            num_classes,
            10,
            dropRate=0.0,
            original=False,
        )
    elif name == "cifar_resnet18_split_bn":
        from .cifar_resnet_split_bn import BasicBlock as BasicBlockSplitBN
        from .cifar_resnet_split_bn import ResNet as ResNetSplitBN

        net = ResNetSplitBN(
            BasicBlockSplitBN, [2, 2, 2, 2], num_classes=num_classes, bn_names=args.bn_names
        )
    elif name == "cifar_wrn_34_10_split_bn":
        from .wrn_split_bn import WideResNet as WideResNetSplitBN

        net = WideResNetSplitBN(
            34,
            num_classes,
            10,
            dropRate=0.0,
            original=False,
            bn_names=args.bn_names,
        )
    else:
        raise NotImplementedError
    return net


def load_model(
    args,
    name,
    num_classes,
    extract_layers=[],
    is_avg_pool=True,
    is_relu=True,
):
    net = get_model(args, name, num_classes)

    if "split_bn" in name:
        net = FeatureExtractorSplitBN(
            net, extract_layers, is_avg_pool=is_avg_pool, is_relu=is_relu, bn_names=args.bn_names
        )
    else:
        net = FeatureExtractor(net, extract_layers, is_avg_pool=is_avg_pool, is_relu=is_relu)
    return net


if __name__ == "__main__":
    net = get_model({}, "cifar_wrn_28_10", 10)
    print(net)
