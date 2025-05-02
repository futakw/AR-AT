import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_weight_distrib_dict(model):
    """
    return a dict of weight distribution for each layer.
    """
    weight_distrib_dict = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weight_distrib_dict[name] = layer.weight.flatten().clone().detach().cpu().numpy()

    return weight_distrib_dict
