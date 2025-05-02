import torch
import numpy as np


def get_batch_stats(features_dict):
    """
    Get batch statistics, for input x.
        Calculate channel mean,std.

    model should output features, for each convolution layer.

    features_dict: {layer_name: feat}
        feat: shape (batch_size, channels, height, width) of conv layer
    channel_mean: shape (channels,)
    channel_std: shape (channels,)
    """
    # print(features_dict.keys())
    # calc channel mean, std
    channel_mean_dict = {}
    channel_std_dict = {}
    for k in features_dict:
        feat = features_dict[k].detach().cpu()
        # print(f"feat.shape: {feat.shape}")
        if len(feat.shape) == 4:
            channel_mean_dict[k] = feat.mean(dim=(0, 2, 3))
            channel_std_dict[k] = feat.std(dim=(0, 2, 3))
        elif len(feat.shape) == 2:
            channel_mean_dict[k] = feat.mean(dim=0)
            channel_std_dict[k] = feat.std(dim=0)
        else:
            raise ValueError(f"Invalid feat shape: {feat.shape}")
    return channel_mean_dict, channel_std_dict
