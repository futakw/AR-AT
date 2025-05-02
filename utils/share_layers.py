import torch
import torch.nn as nn


def get_layer_names(net, start_name):
    """
    get layer names from net, that includes start_name.
    only add the name of the layer, not the name of the parameters.
    """
    layer_names = []
    for name, module in net.named_modules():
        if name.startswith(start_name):
            layer_names.append(name)
    return layer_names


def get_all_layer_names(net):
    """
    get layer names from net, that includes start_name.
    only add the name of the layer, not the name of the parameters.
    """
    layer_names = []
    for name, module in net.named_modules():
        layer_names.append(name)
    return layer_names


def getattr_(obj, attr):
    """if attr is digit, return obj[int(attr)], else return getattr(obj, attr)"""
    if isinstance(obj, nn.Module):
        return getattr(obj, attr)
    elif isinstance(obj, list) and attr.isdigit():
        return obj[int(attr)]


def getattr_recursive(obj, attr_list):
    """if attr is digit, return obj[int(attr)], else return getattr(obj, attr)"""
    for attr in attr_list:
        obj = getattr_(obj, attr)
    return obj


def share_layers(
    net1, net2, share_layer_name_list, separate_bn=False, exclude_layer_name_list=[]
):
    """
    Share layers between net1 and net2. (which means net1 and net2 will share the same parameters in shared memory)

    Args:
        net1: nn.Module
        net2: nn.Module, which should have the same architecture as net1.
        share_layer_name_list: list of layer names to share.
        separate_bn: If True, BN layers will not be shared.
        exclude_layer_name_list: list of layer names to exclude from sharing.

    for example:
        share_layer_name_list = [
            "layer1.0",
        ]
        then, layers with name
            "layer1.0.conv1", "layer1.0.bn1", "layer1.0.conv2",
            "layer1.0.bn2", "layer1.0.shortcut", "layer1.0.bn3"
        will be shared.

        if separate_bn is True,
        then "layer1.0.bn1", "layer1.0.bn2", "layer1.0.bn3" will not be shared.
    """
    print("share_layer_name_list: ", share_layer_name_list)
    assert isinstance(share_layer_name_list, list)
    share_layer_name_list = [name for name in share_layer_name_list if name != ""]
    if exclude_layer_name_list is None:
        exclude_layer_name_list = []
    else:
        exclude_layer_name_list = [
            name for name in exclude_layer_name_list if name != ""
        ]
    SHARE_ALL = True if "all" in share_layer_name_list else False
    print("Share all layers: ", SHARE_ALL)

    def share_layer_1by1(share_layer_name, share_module, net2, shared_layers, separate_bn=False):
        """
        share one layer
        """
        if any(True for _ in share_module.named_children()):
            for l_name, module in share_module.named_children():
                layer_name = (
                    share_layer_name + "." + l_name
                    if share_layer_name != ""
                    else l_name
                )
                share_layer_1by1(layer_name, module, net2, shared_layers, separate_bn=separate_bn)
        else:
            if share_layer_name in shared_layers:  # Check if layer is already shared
                return  # Skip if already shared
            shared_layers.add(share_layer_name)  # Add to set of shared layers
            split_layer_name = share_layer_name.split(".")
            if separate_bn and (
                isinstance(share_module, nn.BatchNorm2d)
                or isinstance(share_module, nn.BatchNorm1d)
                or isinstance(share_module, nn.LayerNorm)
                or isinstance(share_module, nn.GroupNorm)
                or isinstance(share_module, nn.InstanceNorm2d)
            ):
                # print("Skipped: ", share_layer_name, type(share_module))
                return
            setattr(
                getattr_recursive(net2, split_layer_name[:-1]),
                split_layer_name[-1],
                share_module,
            )
            print("shared: ", share_layer_name, type(share_module))

    shared_layers = set()  # Set to keep track of shared layers
    for l_name, module in net1.named_modules():
        if SHARE_ALL or (
            l_name in share_layer_name_list and l_name not in exclude_layer_name_list
        ):
            share_layer_1by1(l_name, module, net2, shared_layers, separate_bn=separate_bn)

    # if share_layer_name_list == ["all"]:
    #     share_layer_name_list = get_all_layer_names(net1)

    # for share_layer_name in share_layer_name_list:
    #     if share_layer_name == "":
    #         continue
    #     all_layer_names = get_layer_names(net1, share_layer_name)

    #     # replace layer with net2's layer (make sure that when net1's weight is updated, net2's weight is also updated, and vice versa)
    #     for l_name in all_layer_names:
    #         if exclude_layer_name_list:
    #             if l_name in exclude_layer_name_list:
    #                 continue
    #         split_layer_name = l_name.split(".")
    #         if split_layer_name[-1].isdigit():
    #             continue
    #         shared_layer = getattr_recursive(net1, split_layer_name)
    #         if separate_bn and isinstance(shared_layer, nn.BatchNorm2d):
    #             continue
    #         setattr(getattr_recursive(net2, split_layer_name[:-1]), split_layer_name[-1], shared_layer)
    #         print("shared: ", l_name)

    return net1, net2


