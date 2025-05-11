import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
from tqdm import tqdm

import attacks


def evaluate(
    net,
    test_loader,
    adv=True,
    max_n=100000,
    lpnorm="inf",
    eps=8.0 / 255,
    alpha=2.0 / 255,
    steps=20,
):
    if lpnorm == "inf":
        adversary = attacks.PGD_linf(epsilon=eps, num_steps=steps, step_size=alpha).cuda()
    elif lpnorm in ["2", 2]:
        adversary = torchattacks.PGDL2(net, eps=eps, alpha=alpha, steps=steps)
    else:
        raise ValueError("lpnorm should be 'inf' or '2'")

    net.eval()
    if adv is False:
        torch.set_grad_enabled(False)
    running_loss = 0
    running_acc = 0
    count = 0
    for i, (bx, by) in tqdm(enumerate(test_loader), total=len(test_loader), desc="eval"):
        bx, by = bx.cuda(), by.cuda()
        count += by.size(0)

        if adv:
            if lpnorm == "inf":
                adv_bx = adversary(net, bx, by)
            elif lpnorm in ["2", 2]:
                adv_bx = adversary(bx, by)
                
            if adv_bx.requires_grad:
                adv_bx = adv_bx.detach()
            with torch.no_grad():
                logits = net(adv_bx)

        else:
            with torch.no_grad():
                logits = net(bx, get_feat=False)

        loss = F.cross_entropy(logits.data, by, reduction="sum")
        running_loss += loss.cpu().data.numpy()
        running_acc += (torch.max(logits, dim=1)[1] == by).float().sum(0).cpu().data.numpy()

        if count > max_n:
            break
    running_loss /= count
    running_acc /= count

    loss = running_loss
    acc = running_acc

    if adv is False:
        torch.set_grad_enabled(True)

    return loss, acc


def evaluate_auto_attack(net, test_loader, n_classes, adv=True, max_n=np.inf, lpnorm="inf"):
    if lpnorm == "inf":
        adversary = torchattacks.AutoAttack(net, norm="Linf", eps=8.0 / 255, n_classes=n_classes)
    elif lpnorm in ["2", 2]:
        adversary = torchattacks.AutoAttack(net, norm="L2", eps=128.0 / 255, n_classes=n_classes)
    else:
        raise ValueError("lpnorm should be 'inf' or '2'")

    net.eval()
    if adv is False:
        torch.set_grad_enabled(False)
    running_loss = 0
    running_acc = 0
    count = 0
    for i, (bx, by) in tqdm(enumerate(test_loader), total=len(test_loader), desc="eval"):
        bx, by = bx.cuda(), by.cuda()
        count += by.size(0)

        if adv:
            adv_bx = x = adversary(bx, by) if adv else bx  ######
            if adv_bx.requires_grad:
                adv_bx = adv_bx.detach()
            with torch.no_grad():
                logits = net(adv_bx)
        else:
            with torch.no_grad():
                logits = net(bx)

        loss = F.cross_entropy(logits.data, by, reduction="sum")
        running_loss += loss.cpu().data.numpy()
        running_acc += (torch.max(logits, dim=1)[1] == by).float().sum(0).cpu().data.numpy()

        if count > max_n:
            break
    running_loss /= count
    running_acc /= count

    loss = running_loss
    acc = running_acc

    if adv is False:
        torch.set_grad_enabled(True)
    return loss, acc
