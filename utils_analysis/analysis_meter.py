import torch
import torch.nn as nn
import numpy as np
import copy

from utils_analysis.grad import (
    get_grad_dict,
    flatten_grad,
    calc_grad_sim_flatten,
    calc_grad_sim_layer,
    calc_model_distance,
    calc_grad_norm,
    count_grad_conflict_ratio,
)
from utils_analysis.batch_stats import get_batch_stats


class AnalysisMeter:
    def __init__(self, model, is_analysis_types=["grad", "batch_stats"]):
        """
        list: stats for each batch
        dict: {name: list (stats for each batch)}
        """
        self.N_SAVE_LAST = 100

        self.init_model = copy.deepcopy(model)
        self.pre_model = self.init_model

        self.pre_grad_dict = None
        self.batch_grad_sim = [1]

        self.pre_grad_flatten_each_loss = {}
        self.batch_grad_sim_each_loss = {}

        self.model_dist = []
        self.model_dist_dict = {}

        # self.model_dist_oneStep = []
        self.model_dist_oneStep_dict = {}
        self.model_dist_oneStep_eachlayer_dict = {}

        self.loss = []
        self.loss_dict = {}

        self.is_feature_analysis = True  # turn off when not needed, since it takes time

        ###################################
        ######## gradient analysis ########
        ###################################

        # Calculate gradient similarity between two loss functions
        # dict: {loss_pair_name: list (stats for each batch)}
        self.multiloss_grad_sim_dict = {}
        self.multiloss_grad_conflict_ratio_dict = {}

        # Calculate layer-wise gradient similarity between two loss functions
        # dict: {loss_pair_name:
        #  {layer_name: list (stats for each batch)}
        # }
        self.multiloss_grad_sim_layerwise_dict = {}

        # dict: {loss_name:
        #  {layer_name: [(C,), (C,), ...]}
        # }
        self.channel_abs_mean_grad_dict = {}
        self.channel_mean_grad_dict = {}
        # self.grad_dict = {}
        self.grad_norm = {}

        ##################################
        ######## batch statistics ########
        ##################################
        # dict: {
        #   input_1: {layer_name: [(C,), (C,), ...]]},
        #   input_2: {layer_name: [(C,), (C,), ...]]},
        #   ...
        # }
        self.channel_mean_dict = {}
        self.channel_std_dict = {}

        # batch statistics analysis: running_mean, running_var
        self.running_mean_dict = {}
        self.running_var_dict = {}

    def gradient_analysis(self, model, loss_dict):
        """
        Gradient analysis.
        """
        if len(loss_dict) == 0:
            return
        else:
            for k in loss_dict:  # for bug fix
                loss_dict[k] += 1e-20
            loss = sum(loss_dict.values())

            for loss_key in ["sum"] + list(loss_dict.keys()):
                if loss_key == "sum":
                    loss = sum(loss_dict.values())
                else:
                    loss = loss_dict[loss_key]

                this_grad_dict = get_grad_dict(model, loss)
                this_grad_flatten = flatten_grad(this_grad_dict)

                # calc grad magnitude for each layer, each channel
                # grad: (B, C, H, W)
                # channel_abs_mean_grad_dict: {loss_name: {layer_name: [(C,), (C,), ...]}}
                # channel_mean_grad_dict: {loss_name: {layer_name: [(C,), (C,), ...]}}
                # grad_dict: {loss_name: {layer_name: [(B, C, H, W), (B, C, H, W), ...]}}
                for layer_name in this_grad_dict:
                    this_grad = this_grad_dict[layer_name]
                    # assert (
                    #     len(this_grad.shape) == 4
                    # ), f"{layer_name}: this_grad.shape should be (B, C, H, W), but {this_grad.shape} is given."
                    if len(this_grad.shape) != 4:
                        # only conv layer
                        continue
                    self.channel_abs_mean_grad_dict.setdefault(loss_key, {}).setdefault(
                        layer_name, []
                    ).append(this_grad.abs().mean(dim=(0, 2, 3)).detach().cpu())
                    self.channel_mean_grad_dict.setdefault(loss_key, {}).setdefault(layer_name, []).append(
                        this_grad.mean(dim=(0, 2, 3)).detach().cpu()
                    )
                    # self.grad_dict.setdefault(loss_key, {}).setdefault(layer_name, []).append(this_grad.detach().cpu())

                # calc grad norm
                self.grad_norm.setdefault(loss_key, []).append(calc_grad_norm(this_grad_flatten))


                # calc batch grad similarity
                if loss_key not in self.batch_grad_sim_each_loss:
                    self.batch_grad_sim_each_loss[loss_key] = [1]
                if loss_key == "sum":
                    if self.pre_grad_dict is not None:
                        pre_grad_flatten = flatten_grad(self.pre_grad_dict)
                        self.batch_grad_sim.append(
                            calc_grad_sim_flatten(pre_grad_flatten, this_grad_flatten)
                        )
                elif loss_key in self.pre_grad_flatten_each_loss:
                    pre_grad_flatten = self.pre_grad_flatten_each_loss[loss_key]
                    self.batch_grad_sim_each_loss[loss_key].append(
                        calc_grad_sim_flatten(pre_grad_flatten, this_grad_flatten)
                    )

                self.pre_grad_flatten_each_loss[loss_key] = this_grad_flatten

            # calc grad similarity
            len_loss_dict = len(loss_dict)
            if len_loss_dict == 1:
                pass
            elif len_loss_dict >= 2:
                keys = list(loss_dict.keys())
                loss1 = loss_dict[keys[0]]
                loss2 = loss_dict[keys[1]]
                self.loss_dict.setdefault(keys[0], []).append(loss1.item())
                self.loss_dict.setdefault(keys[1], []).append(loss2.item())
                if len_loss_dict == 3:
                    loss3 = loss_dict[keys[2]]
                    self.loss_dict.setdefault(keys[2], []).append(loss3.item())
                else:
                    loss3 = None

                # 12
                loss_pair_name = f"{keys[0]}-{keys[1]}"
                grad_dict_1 = get_grad_dict(model, loss1)
                grad_dict_2 = get_grad_dict(model, loss2)

                layerwise_grad_sim_dict = calc_grad_sim_layer(grad_dict_1, grad_dict_2)
                self.multiloss_grad_sim_layerwise_dict.setdefault(loss_pair_name, []).append(
                    layerwise_grad_sim_dict
                )

                grad_flatten_1 = flatten_grad(grad_dict_1)
                grad_flatten_2 = flatten_grad(grad_dict_2)
                grad_sim = calc_grad_sim_flatten(grad_flatten_1, grad_flatten_2)
                self.multiloss_grad_sim_dict.setdefault(loss_pair_name, []).append(grad_sim)

                # clac grad conflict ratio
                grad_conflict_ratio = count_grad_conflict_ratio(grad_flatten_1, grad_flatten_2)
                self.multiloss_grad_conflict_ratio_dict.setdefault(loss_pair_name, []).append(
                    grad_conflict_ratio
                )

                if loss3 is not None:
                    # 23
                    loss_pair_name = f"{keys[1]}-{keys[2]}"
                    grad_dict_3 = get_grad_dict(model, loss3)
                    layerwise_grad_sim_dict = calc_grad_sim_layer(grad_dict_2, grad_dict_3)
                    self.multiloss_grad_sim_layerwise_dict.setdefault(loss_pair_name, []).append(
                        layerwise_grad_sim_dict
                    )

                    grad_flatten_3 = flatten_grad(grad_dict_3)
                    grad_sim = calc_grad_sim_flatten(grad_flatten_2, grad_flatten_3)
                    self.multiloss_grad_sim_dict.setdefault(loss_pair_name, []).append(grad_sim)

                    # 31
                    loss_pair_name = f"{keys[2]}-{keys[0]}"
                    layerwise_grad_sim_dict = calc_grad_sim_layer(grad_dict_3, grad_dict_1)
                    self.multiloss_grad_sim_layerwise_dict.setdefault(loss_pair_name, []).append(
                        layerwise_grad_sim_dict
                    )

                    self.multiloss_grad_sim_dict.setdefault(loss_pair_name, []).append(grad_sim)

            self.loss.append(loss.item())
            self.pre_grad_dict = this_grad_dict

    def update(
        self,
        model,
        features_dict=None,
        loss_dict=None,
        lr=None,
    ):
        """
        AnalysisMeter.update

        # batch statistics analysis
            features_dict: dict of features for each modality.
                e.g. {"x": {layername: feat, ...}, "x_adv": {layername: feat, ...}}

        # gradient analysis
            loss: total loss
            loss_dict: dict of loss for each role.
                e.g. {"loss1": loss1, "loss2": loss2, "loss3": loss3}
        """
        # important:
        #  copy model to avoid updating the original model accidentally.
        #  set model to eval mode
        # print("AnalysisMeter.update")

        # calc model distance
        model_dist, model_dist_dict = calc_model_distance(self.init_model, model)
        self.model_dist.append(model_dist)
        for k in model_dist_dict:
            self.model_dist_dict.setdefault(k, []).append(model_dist_dict[k])

        # calc model distance (one step)
        for loss_key in ["sum"] + list(loss_dict.keys()):
            if loss_key == "sum":
                loss = sum(loss_dict.values())
            else:
                loss = loss_dict[loss_key]
            # print("loss_key", loss_key, loss)
            loss.backward(retain_graph=True)
            grad_dict = {}
            for name, layer in model.named_modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if layer.weight.grad is not None:
                        grad_dict[name] = layer.weight.grad.clone()
                        # zero grad
                        layer.weight.grad.zero_()
                # else:
                #     print("Not Conv or Linear: ", name)

            updates_flatten = None
            model_updated_with_this_loss = copy.deepcopy(model)
            for name, layer in model_updated_with_this_loss.named_modules():
                if name in grad_dict:
                    layer.weight.data -= lr * grad_dict[name]
                    # if "layer4.0.conv2" in name:
                    #     print("name", name, "norm update:", torch.norm(lr * grad_dict[name]))
                    #     print((lr * grad_dict[name]).flatten().shape)
                    # print((lr * grad_dict[name]).flatten()[:10])

                    if updates_flatten is None:
                        updates_flatten = lr * grad_dict[name].flatten()
                    else:
                        updates_flatten = torch.cat([updates_flatten, lr * grad_dict[name].flatten()])

            updates_flatten_norm = torch.norm(updates_flatten)
            # print("updates_flatten:", updates_flatten)
            # print("updates_flatten_norm:", updates_flatten_norm)

            model_dist_oneStep, model_dist_dict_oneStep = calc_model_distance(
                model, model_updated_with_this_loss
            )
            # print("model_dist_oneStep: ", model_dist_oneStep)
            self.model_dist_oneStep_dict.setdefault(loss_key, []).append(model_dist_oneStep)
            for k in model_dist_dict_oneStep:
                self.model_dist_oneStep_eachlayer_dict.setdefault(loss_key, {}).setdefault(k, []).append(
                    model_dist_dict_oneStep[k]
                )

        # batch statistics analysis
        if self.is_feature_analysis:
            if features_dict is not None:
                # print("features_dict", features_dict.keys())
                for input_name in features_dict:
                    # print("-", input_name)
                    assert input_name in [
                        "x",
                        "adv_x",
                    ], f"input_name should be in ['x', 'adv_x'], but {input_name} is given."
                    this_features_dict = features_dict[input_name]
                    # print("this_features_dict", this_features_dict.keys())
                    channel_mean_dict, channel_std_dict = get_batch_stats(this_features_dict)
                    # print("channel_mean_dict", channel_mean_dict.keys())

                    for k in channel_mean_dict:
                        # print(k, input_name)
                        self.channel_mean_dict.setdefault(k, {}).setdefault(input_name, []).append(
                            channel_mean_dict[k].detach().cpu()
                        )
                        self.channel_std_dict.setdefault(k, {}).setdefault(input_name, []).append(
                            channel_std_dict[k].detach().cpu()
                        )

        # batch statistics analysis: running_mean, running_var
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # print(f"{name} - Running Mean: {module.running_mean}, Running Var: {module.running_var}")
                self.running_mean_dict.setdefault(name, []).append(module.running_mean.detach().cpu())
                self.running_var_dict.setdefault(name, []).append(module.running_var.detach().cpu())

        # gradient analysis
        self.gradient_analysis(model, loss_dict)

        # print("batch_grad_sim:", self.batch_grad_sim[-1])
        # print("multiloss_grad_sim:", self.multiloss_grad_sim_dict)

        # update pre_model
        self.pre_model = copy.deepcopy(model)

        # limit the number of saved stats
        metrics = self.get_metrics()
        for k, v in metrics.items():
            if isinstance(v, list) and len(v) > self.N_SAVE_LAST:
                self.__dict__[k] = v[-self.N_SAVE_LAST :]
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, list) and len(v2) > self.N_SAVE_LAST:
                        self.__dict__[k][k2] = v2[-self.N_SAVE_LAST :]
                    elif isinstance(v2, dict):
                        for k3, v3 in v2.items():
                            if isinstance(v3, list) and len(v3) > self.N_SAVE_LAST:
                                self.__dict__[k][k2][k3] = v3[-self.N_SAVE_LAST :]
                            else:
                                pass
            else:
                pass

    def get_metrics(self, verbose=False):
        # get metrics
        d = {
            "model_dist": self.model_dist,
            "model_dist_dict": self.model_dist_dict,
            "model_dist_oneStep_dict": self.model_dist_oneStep_dict,
            "model_dist_oneStep_eachlayer_dict": self.model_dist_oneStep_eachlayer_dict,
            "loss": self.loss,
            "loss_dict": self.loss_dict,
            "batch_grad_sim": self.batch_grad_sim,
            "batch_grad_sim_each_loss": self.batch_grad_sim_each_loss,
            "multiloss_grad_sim_dict": self.multiloss_grad_sim_dict,
            "multiloss_grad_conflict_ratio_dict": self.multiloss_grad_conflict_ratio_dict,
            "multiloss_grad_sim_layerwise_dict": self.multiloss_grad_sim_layerwise_dict,
            "channel_abs_mean_grad_dict": self.channel_abs_mean_grad_dict,
            "channel_mean_grad_dict": self.channel_mean_grad_dict,
            "grad_norm": self.grad_norm,
            "channel_mean_dict": self.channel_mean_dict,
            "channel_std_dict": self.channel_std_dict,
            "running_mean_dict": self.running_mean_dict,
            "running_var_dict": self.running_var_dict,
        }

        if verbose:
            for k, v in self.get_metrics().items():
                if isinstance(v, list) and len(v) > 0:
                    print(k, v[-1])
                elif isinstance(v, float):
                    print(k, v)
                elif isinstance(v, dict):
                    if k == "multiloss_grad_sim_dict":
                        for k2, v2 in v.items():
                            print(k, k2, v2[-1])
                else:
                    pass
        return d
