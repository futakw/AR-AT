import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """
    Converts a model into a feature extractor, which returns logits and dict of features.

    Usage:
        model = get_model(args, name, num_classes)
        hook_layers = ["layer1", "layer2", "layer3", "layer4"]
        feature_extractor = FeatureExtractor(model, hook_layers)
        out, features = feature_extractor(x, get_feat=True) # out: logits, features: dict of features

    Args:
        model: nn.Module
        hook_layers: list of layer names to extract features
        is_return_dict: If True, return dict of features.
        is_return_logits: If True, return logits.
        is_avg_pool: If True, apply avg_pool to feature map.
        is_relu: If True, apply relu to feature map.

    """

    def __init__(
        self,
        model: nn.Module,
        hook_layers: list,
        is_return_dict=True,
        is_return_logits=True,
        is_avg_pool=False,
        is_relu=True,
    ):
        super(FeatureExtractor, self).__init__()
        self.is_return_dict = is_return_dict
        self.is_avg_pool = is_avg_pool
        self.is_return_logits = is_return_logits
        self.is_relu = is_relu

        self.model = copy.deepcopy(model)
        self.hook_layers_names = hook_layers
        self.hook_handles = []  # Store hook handles
        self.set_hook_layers()  # Set hook layers

        self.features = {}

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def set_hook_layers(self):
        self.remove_hooks()  # reset

        hook_layers = self.hook_layers_names
        if "all" in hook_layers:
            hook_layers = [name for name, module in self.model.named_modules()]
            hook_layers = hook_layers[1:]  # remove first layer
        hook_layers = [name for name in hook_layers if name != ""]

        self.hook_layers = hook_layers
        self.hook_layers_dict = {k: k for k in hook_layers}
        print("Set hook_layers:", hook_layers)

        added_layer_names = []
        for name, module in self.model.named_modules():
            if name in self.hook_layers:
                handle = module.register_forward_hook(self.extract(name=name))
                self.hook_handles.append(handle)
                added_layer_names += [name]

        print("Added layers:", added_layer_names)
        assert len(added_layer_names) == len(
            hook_layers
        ), f"Some layer did not exist. {set(added_layer_names) - set(hook_layers)},{set(hook_layers) - set(added_layer_names)}"

    def avg_pool_feature(self, o):
        """o: cpu feature map"""
        if len(o.shape) == 4:
            feat = F.adaptive_avg_pool2d(o, (1, 1)).reshape(o.shape[0], -1)
        elif len(o.shape) == 3:
            feat = torch.mean(o, 1)
        elif len(o.shape) == 2:
            feat = o
        else:
            print(o.shape)
            raise ValueError
        return feat

    def extract(self, name=""):
        def _extract(module, f_in, f_out, name=name):
            if len(f_out) == 2: # for split bn model
                f_out = f_out[0]

            if self.is_relu:
                f_out = F.relu(f_out)
            if self.is_avg_pool:
                f_out = self.avg_pool_feature(f_out)
            self.features[name] = f_out
            # print(f"extracted: {name}", f_out.shape, len(self.features))

        return _extract

    def forward(self, input, get_feat=False, **kwargs):
        out = self.model(input, **kwargs)

        if get_feat == False:
            self.features = {}
            return out
        assert len(self.features) == len(self.hook_layers), (
            "Something's wrong.",
            len(self.features),
            len(self.hook_layers),
            set(self.features) - set(self.hook_layers),
            set(self.hook_layers) - set(self.features),
        )
        d = {self.hook_layers_dict[k]: feat for k, feat in self.features.items()}
        self.features = {}
        if self.is_return_logits:
            return out, d
        else:
            return d



class FeatureExtractorSplitBN(nn.Module):
    """
    Converts a model into a feature extractor, which returns logits and dict of features.

    Usage:
        model = get_model(args, name, num_classes)
        hook_layers = ["layer1", "layer2", "layer3", "layer4"]
        feature_extractor = FeatureExtractor(model, hook_layers)
        out, features = feature_extractor(x, get_feat=True) # out: logits, features: dict of features

    Args:
        model: nn.Module
        hook_layers: list of layer names to extract features
        is_return_dict: If True, return dict of features.
        is_return_logits: If True, return logits.
        is_avg_pool: If True, apply avg_pool to feature map.
        is_relu: If True, apply relu to feature map.

    """

    def __init__(
        self,
        model: nn.Module,
        hook_layers: list,
        is_return_dict=True,
        is_return_logits=True,
        is_avg_pool=False,
        is_relu=True,
        bn_names=["base"],
    ):
        super(FeatureExtractorSplitBN, self).__init__()
        self.is_return_dict = is_return_dict
        self.is_avg_pool = is_avg_pool
        self.is_return_logits = is_return_logits
        self.is_relu = is_relu
        self.bn_names = bn_names    

        self.model = copy.deepcopy(model)
        self.hook_layers_names = hook_layers
        self.hook_handles = []  # Store hook handles
        self.set_hook_layers()  # Set hook layers

        self.features = {}

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def set_hook_layers(self):
        self.remove_hooks()  # reset

        hook_layers = self.hook_layers_names
        if "all" in hook_layers:
            hook_layers = [name for name, module in self.model.named_modules()]
            hook_layers = hook_layers[1:]  # remove first layer
        hook_layers = [name for name in hook_layers if name != ""]

        self.hook_layers_num = len(hook_layers)
        # split bn
        _hook_layers = []
        for name in hook_layers:
            if "bn" in name:
                for bn_name in self.bn_names:
                    _hook_layers.append(f"{name}.{bn_name}")
            else:
                _hook_layers.append(name)
        self.hook_layers = _hook_layers

        # self.hook_layers = hook_layers
        self.hook_layers_dict = {k: k for k in self.hook_layers}
        print("Set hook_layers:", hook_layers)

        added_layer_names = []
        for name, module in self.model.named_modules():
            if name in self.hook_layers:
                handle = module.register_forward_hook(self.extract(name=name))
                self.hook_handles.append(handle)
                added_layer_names += [name]

        print("Added layers:", added_layer_names)
        assert len(added_layer_names) == len(
            self.hook_layers
        ), f"Some layer did not exist. {set(added_layer_names) - set(hook_layers)},{set(hook_layers) - set(added_layer_names)}"

    def avg_pool_feature(self, o):
        """o: cpu feature map"""
        if len(o.shape) == 4:
            feat = F.adaptive_avg_pool2d(o, (1, 1)).reshape(o.shape[0], -1)
        elif len(o.shape) == 3:
            feat = torch.mean(o, 1)
        elif len(o.shape) == 2:
            feat = o
        else:
            print(o.shape)
            raise ValueError
        return feat

    def extract(self, name=""):
        def _extract(module, f_in, f_out, name=name):
            if len(f_out) == 2: # for split bn model
                f_out = f_out[0]

            if self.is_relu:
                f_out = F.relu(f_out)
            if self.is_avg_pool:
                f_out = self.avg_pool_feature(f_out)
            self.features[name] = f_out
            # print(f"extracted: {name}", f_out.shape, len(self.features))

        return _extract

    def forward(self, input, get_feat=False, bn_name=None):
        if bn_name is None:
            bn_name = self.model.bn_name
        out = self.model(input, bn_name=bn_name)

        if get_feat == False:
            self.features = {}
            return out
        assert len(self.features) == self.hook_layers_num, (
            "Something's wrong.",
            len(self.features),
            len(self.hook_layers),
            set(self.features) - set(self.hook_layers),
            set(self.hook_layers) - set(self.features),
        )
        d = {self.hook_layers_dict[k]: feat for k, feat in self.features.items()}
        d = {k.replace(f".{bn_name}", ""): v for k, v in d.items()} # remove bn_name
        self.features = {}
        if self.is_return_logits:
            return out, d
        else:
            return d
        
    def named_parameters(self, *args, **kwargs):
        return self.model.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)