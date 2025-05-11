import torch
from collections import OrderedDict

def convert_splitbn_to_standard(splitbn_model, standard_model, bn_name="clean"):
    split_state = splitbn_model.state_dict()
    standard_state = standard_model.state_dict()
    new_state = OrderedDict()

    for key in standard_state.keys():
        if key in split_state:
            # Directly shared parameters (e.g., conv layers, linear layer)
            new_state[key] = split_state[key]
        else:
            # BatchNorm parameters in split version are under ModuleDict
            split_key = insert_bn_name(key, bn_name)
            if split_key in split_state:
                new_state[key] = split_state[split_key]
            else:
                print(f"Warning: key {key} not found in split model, skipping")

    standard_model.load_state_dict(new_state)
    return standard_model


def insert_bn_name(key, bn_name):
    parts = key.split(".")

    # Handle top-level BN layer (used in both ResNet and WideResNet)
    if parts[0] == "bn1":
        return f"{parts[0]}.{bn_name}." + ".".join(parts[1:])

    # Handle ResNet-style layers: layerX.Y.bnZ.* or layerX.Y.shortcut.*
    if parts[0].startswith("layer"):
        if parts[2].startswith("bn"):
            # e.g., layer1.0.bn1.weight → layer1.0.bn1.base.weight
            return f"{parts[0]}.{parts[1]}.{parts[2]}.{bn_name}." + ".".join(parts[3:])
        if parts[2] == "shortcut":
            if parts[3] == "0":
                # e.g., layer1.0.shortcut.0.weight → layer1.0.shortcut.weight
                return f"{parts[0]}.{parts[1]}.shortcut." + ".".join(parts[4:])
            if parts[3] == "1":
                # e.g., layer1.0.shortcut.1.weight → layer1.0.shortcut_bn.base.weight
                return f"{parts[0]}.{parts[1]}.shortcut_bn.{bn_name}." + ".".join(parts[4:])

    # Handle WideResNet-style blocks: blockX.layer.Y.bnZ.* → blockX.layer.Y.bnZ.base.*
    if parts[0].startswith("block") and parts[1] == "layer":
        return f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}.{bn_name}." + ".".join(parts[4:])

    # Return None if no matching rule
    return None


if __name__ == "__main__":
    from cifar_resnet import ResNet, BasicBlock
    from cifar_resnet_split_bn import ResNet as ResNetSplitBN
    from cifar_resnet_split_bn import BasicBlock as BasicBlockSplitBN

    split_model = ResNetSplitBN(BasicBlockSplitBN, [2, 2, 2, 2], num_classes=10, bn_names=["clean", "adv"])
    standard_model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    bn_name = "adv"

    path = "/home/fmg2/waseda/AR-AT/ckpts/cifar10/cifar_resnet18_split_bn/ARAT/seed0/model_last.pt"
    split_model.load_state_dict(torch.load(path))

    
    converted_model = convert_splitbn_to_standard(split_model, standard_model, bn_name)
    print("Successfully converted split BN model to standard model.")


    # WRN
    from wrn_split_bn import WideResNet as WideResNetSplitBN
    from wrn_split_bn import BasicBlock as BasicBlockSplitBN
    from wrn_split_bn import NetworkBlock as NetworkBlockSplitBN
    from wrn import WideResNet as WideResNet
    from wrn import BasicBlock as BasicBlock
    from wrn import NetworkBlock as NetworkBlock
    
    split_model = WideResNetSplitBN(34, 10, 10, dropRate=0.0, original=False, bn_names=["clean", "adv"])
    standard_model = WideResNet(34, 10, 10, dropRate=0.0, original=False)
    bn_name = "adv"
    path = "/home/fmg2/waseda/AR-AT/ckpts/cifar10/cifar_wrn_34_10_split_bn/ARAT/seed0/model_last.pt"
    split_model.load_state_dict(torch.load(path))
    converted_model = convert_splitbn_to_standard(split_model, standard_model, bn_name)
    print("Successfully converted split BN model to standard model.")