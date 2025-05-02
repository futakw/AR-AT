import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_grad_dict(model, loss):
    """
    return a dict of gradients for each layer.
    """
    loss.backward(retain_graph=True)
    grad_dict = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if layer.weight.grad is not None:
                grad_dict[name] = layer.weight.grad.clone()
                # zero grad
                layer.weight.grad.zero_()

    return grad_dict


def flatten_grad(grad_dict):
    """
    concat all flatten grads from each layer.
    """
    return torch.cat([grad.flatten() for grad in grad_dict.values()])


def calc_grad_sim_flatten(
    grad_flatten_1,
    grad_flatten_2,
):
    """
    calculate similarity of grads between two loss functions.
    """
    # calc cosine similarity
    cos_sim = F.cosine_similarity(
        grad_flatten_1,
        grad_flatten_2,
        dim=0,
    )
    return cos_sim.item()


def count_grad_conflict_ratio(grad_flatten_1, grad_flatten_2):
    """
    count the ratio of conflicting gradients.
    """
    grad_conflict = (grad_flatten_1 * grad_flatten_2 < 0).sum().item()
    grad_total = grad_flatten_1.numel()
    return grad_conflict / grad_total


def calc_grad_sim_layer(grad_dict_1, grad_dict_2):
    layerwise_grad_sim_dict = {}
    for k in grad_dict_1:
        if k in grad_dict_2:
            cos_sim = F.cosine_similarity(
                grad_dict_1[k].flatten(),
                grad_dict_2[k].flatten(),
                dim=0,
            )
            layerwise_grad_sim_dict[k] = cos_sim.detach().cpu()
    return layerwise_grad_sim_dict


def calc_model_distance(model1, model2):
    """
    ||W1 - W2||_F
    """
    model1_dict = model1.state_dict()
    model2_dict = model2.state_dict()
    # only conv layers

    diff = []
    # diff_dict = {}
    dist_dict = {}
    for k in model1_dict:
        # print(k)
        _diff = (model1_dict[k] - model2_dict[k]).flatten().detach().cpu()
        # diff.append(_diff)
        # diff_dict[k] = _diff
        # print(k, _diff)
        if "conv" in k or "linear" in k:
            if len(_diff) > 1:  # if not scalar
                diff.append(_diff)
                dist_dict[k] = torch.norm(_diff)
                # if "layer4.0.conv2" in k:
                #     print(k, torch.norm(_diff))
                #     print(_diff.shape)
                # print(_diff)
        # else:
        #     print(k, "not conv or linear")
    # print("torch.cat(diff): ", torch.cat(diff))
    # print(torch.norm(torch.cat(diff)))

    dist = torch.norm(torch.cat(diff))
    return dist.item(), dist_dict


def calc_grad_norm(grad_flatten):
    """
    calculate norm of grads.
    """
    return torch.norm(grad_flatten).item()


def calc_weight_sparsity(model):
    """
    calculate weight sparsity of a model.

    """
    model_dict = model.state_dict()
    total = 0
    nonzero = 0
    for k in model_dict:
        total += model_dict[k].numel()
        nonzero += (model_dict[k] != 0).sum().item()
    return 1 - nonzero / total


def calc_grad_sim_layer(
    grad_dict_1,
    grad_dict_2,
):
    """
    calculate similarity of grads between two loss functions.
    """
    # calc cosine similarity for each layer
    batch_grad_sim_dict = {}
    for k in grad_dict_1:
        if k in grad_dict_2:
            # average for batch
            cos_sim = F.cosine_similarity(
                grad_dict_1[k].flatten(),
                grad_dict_2[k].flatten(),
                dim=0,
            )
            batch_grad_sim_dict[k] = cos_sim.detach().cpu()
    return batch_grad_sim_dict


def calc_grad_sim_neuron(
    grad_dict_1,
    grad_dict_2,
):
    """
    calculate similarity of grads between two loss functions
    similarity is calculated for each neuron.
    """
    # calc neuron-level cosine similarity
    batch_neuron_grad_sim_dict = {}
    for k in grad_dict_1:
        if k in grad_dict_2:
            cos_sim = F.cosine_similarity(
                grad_dict_1[k].flatten().unsqueeze(0),
                grad_dict_2[k].flatten().unsqueeze(0),
                dim=0,
            )
            batch_neuron_grad_sim_dict[k] = cos_sim.detach().cpu()
    return batch_neuron_grad_sim_dict
