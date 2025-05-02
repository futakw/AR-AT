import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
from tqdm import tqdm

import attacks


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def evaluate(
    args,
    net,
    test_loader,
    adv=True,
    max_n=10000,
    is_analysis=False,
    net_analysis=None,
    lpnorm="inf",
    eps=8.0 / 255,
    alpha=2.0 / 255,
    steps=20,
):
    if eps != 8.0 / 255:
        print("Evaluating for different eps:", eps)
        adversary = attacks.PGD_linf(epsilon=eps, num_steps=steps, step_size=alpha).cuda()
    elif lpnorm == "inf":
        # PGD-20
        adversary = attacks.PGD_linf(epsilon=8.0 / 255, num_steps=20, step_size=2.0 / 255).cuda()
    elif lpnorm in ["2", 2]:
        # PGD-20
        adversary = torchattacks.PGDL2(net, eps=128.0 / 255, alpha=32.0 / 255, steps=20)
    else:
        raise ValueError("lpnorm should be 'inf' or '2'")

    net.eval()
    if adv is False:
        torch.set_grad_enabled(False)
    running_loss = 0
    running_acc = 0
    count = 0
    perturbations = []
    feat_dict_all = {}
    clean_feat_dict_all = {}
    logits_all = []
    analysis_logits_all = []
    labels_all = []
    for i, (bx, by) in tqdm(enumerate(test_loader), total=len(test_loader), desc="eval"):
        bx, by = bx.cuda(), by.cuda()
        count += by.size(0)

        if adv:
            if lpnorm == "inf":
                adv_bx = adversary(net, bx, by)
            elif lpnorm in ["2", 2]:
                adv_bx = adversary(bx, by)
            else:
                raise ValueError("lpnorm should be 'inf' or '2'")
            if adv_bx.requires_grad:
                adv_bx = adv_bx.detach()
            with torch.no_grad():
                logits, feat_dict = net(adv_bx, get_feat=True)
                clean_logits, clean_feat_dict = net(bx, get_feat=True)

            perturbations.append((adv_bx - bx).cpu().detach())
            for k in feat_dict:
                if k not in feat_dict_all:
                    feat_dict_all[k] = []
                feat_dict_all[k].append(feat_dict[k].cpu().detach())
            for k in clean_feat_dict:
                if k not in clean_feat_dict_all:
                    clean_feat_dict_all[k] = []
                clean_feat_dict_all[k].append(clean_feat_dict[k].cpu().detach())
        else:
            with torch.no_grad():
                logits = net(bx, get_feat=False)

        logits_all.append(logits.cpu().detach())
        labels_all.append(by.cpu().detach())

        if is_analysis:
            if net_analysis is not None:
                if adv:
                    with torch.no_grad():
                        analysis_logits, _ = net_analysis(adv_bx, get_feat=True)
                else:
                    with torch.no_grad():
                        analysis_logits, _ = net_analysis(bx, get_feat=True)
                analysis_logits_all.append(analysis_logits.cpu().detach())
            else:
                analysis_logits_all.append(torch.zeros_like(logits).cpu().detach())

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

    if is_analysis:
        perturbations = torch.cat(perturbations, dim=0)
        logits_all = torch.cat(logits_all, dim=0)
        if is_analysis:
            if net_analysis is not None:
                analysis_logits_all = torch.cat(analysis_logits_all, dim=0)
            else:
                analysis_logits_all = torch.zeros_like(logits_all)
        labels_all = torch.cat(labels_all, dim=0)
        for k in feat_dict_all:
            feat_dict_all[k] = torch.cat(feat_dict_all[k], dim=0)
        for k in clean_feat_dict_all:
            clean_feat_dict_all[k] = torch.cat(clean_feat_dict_all[k], dim=0)
        return (
            loss,
            acc,
            perturbations,
            feat_dict_all,
            clean_feat_dict_all,
            logits_all,
            labels_all,
            analysis_logits_all,
        )
    return loss, acc


def eval_feature_invariance(args, net, layer_name, test_loader, adversary, max_n=10000):
    """
    return average cosine similarity between adversarial and clean features
    """
    net.eval()

    cos_sim_sum = 0
    count = 0
    for i, (bx, by) in tqdm(enumerate(test_loader), total=len(test_loader), desc="eval"):
        bx, by = bx.cuda(), by.cuda()
        count += by.size(0)

        try:
            adv_bx = adversary(bx, by)
        except:
            adv_bx = adversary(net, bx, by)
        if adv_bx.requires_grad:
            adv_bx = adv_bx.detach()

        with torch.no_grad():
            logits, feat_dict = net(adv_bx, get_feat=True)
            feat = feat_dict[layer_name]
            if len(feat.shape) == 4:
                feat = F.avg_pool2d(feat, feat.shape[2:])
                feat = feat.view(feat.shape[0], -1)
            feat = feat / feat.norm(dim=1).view(-1, 1)
            feat_adv = feat
        with torch.no_grad():
            logits, feat_dict = net(bx, get_feat=True)
            feat = feat_dict[layer_name]
            if len(feat.shape) == 4:
                feat = F.avg_pool2d(feat, feat.shape[2:])
                feat = feat.view(feat.shape[0], -1)
            feat = feat / feat.norm(dim=1).view(-1, 1)
            feat_clean = feat

        cos_sim = F.cosine_similarity(feat_adv, feat_clean, dim=1)
        cos_sim = cos_sim.mean()
        cos_sim_sum += cos_sim.cpu().data.numpy() * bs

        if count > max_n:
            break
    avg_cos_sim = cos_sim_sum / count
    print("Average cosine similarity (adv. vs. clean feats): {:.4f}".format(avg_cos_sim))
    return avg_cos_sim


def evaluate_auto_attack(args, net, test_loader, adv=True, max_n=np.inf, lpnorm="inf"):
    if lpnorm == "inf":
        adversary = torchattacks.AutoAttack(net, norm="Linf", eps=8.0 / 255, n_classes=args.num_classes)
    elif lpnorm in ["2", 2]:
        adversary = torchattacks.AutoAttack(net, norm="L2", eps=128.0 / 255, n_classes=args.num_classes)
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


def evaluate_multi(net, test_loader, attack_dict, class_num, max_n=10000):
    from utils.utils import AverageMeter

    class_wise_matrix_dict = {att: np.zeros((class_num, class_num)) for att in attack_dict}
    # to have the same augmentation
    torch_fix_seed()

    net.eval()
    loss_meters = {attack: AverageMeter() for attack in attack_dict}
    acc_meters = {attack: AverageMeter() for attack in attack_dict}
    count = 0
    for batch_idx, (bx, by) in tqdm(enumerate(test_loader), total=len(test_loader), desc="eval"):
        bx, by = bx.cuda(), by.cuda()
        bs = by.size(0)
        count += bs

        for name, adversary in attack_dict.items():
            if name == "natural":
                adv_bx = bx
            else:
                try:
                    adv_bx = adversary(bx, by)
                except:
                    adv_bx = adversary(net, bx, by)
            with torch.no_grad():
                logits = net(adv_bx)

            loss = F.cross_entropy(logits.data, by, reduction="sum")
            loss_meters[name].update(loss.cpu().data.numpy() / bs, bs)
            preds = torch.max(logits, dim=1)[1]
            acc = (preds == by).float().sum(0).cpu().data.numpy()
            acc_meters[name].update(
                acc / bs,
                bs,
            )

            preds_np = preds.detach().cpu().numpy()
            by_np = by.detach().cpu().numpy()
            for i, j in zip(preds_np, by_np):
                class_wise_matrix_dict[name][j][i] += 1

        if count >= max_n:
            break
    torch.set_grad_enabled(True)
    return loss_meters, acc_meters, class_wise_matrix_dict


if __name__ == "__main__":
    import os
    import argparse
    from utils.data import get_data_simple
    from models.load_model import load_model

    from utils.config_utils import str2bool as s2b
    from utils.config_utils import str2strlist as s2sl
    from utils.config_utils import str2intlist as s2il

    ############################################
    ################### Args ###################
    ############################################
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter)
    # seed
    parser.add_argument("--seed", type=int, default=42)
    # model
    parser.add_argument("--model", type=str, default="cifar_resnet18")
    # dataset
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--test_bs", type=int, default=256)

    parser.add_argument("--load_ckpt", type=str, default="")

    parser.add_argument("--log_dir", type=str, default="logs/eval/")
    parser.add_argument("--mark", type=str, default="")

    parser.add_argument(
        "--bn_names",
        type=s2sl,
        default=["base", "base_adv"],
        help="BN layer names to split",
    )
    parser.add_argument("--is_update_bn", type=s2b, default=False)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_fix_seed(args.seed)

    ############################################
    ################### Data ###################
    ############################################
    train_loader, test_loader, num_classes = get_data_simple(args)
    args.num_classes = num_classes

    ############################################
    ############# Create Models ################
    ############################################
    net = load_model(
        args,
        args.model,
        num_classes,
        extract_layers=[],
        is_avg_pool=True,
        is_relu=True,
    )
    net = net.model  # remove feature extractor

    # load checkpoint
    ckpt = torch.load(args.load_ckpt, map_location="cpu")
    try:
        net.load_state_dict(ckpt)
    except:
        # remove model prefix
        ckpt = {k.replace("model.", ""): v for k, v in ckpt.items()}
        net.load_state_dict(ckpt)
    net = net.to(device)
    print("load ckpt:", args.load_ckpt)

    # set bn names
    if "split_bn" in args.model:
        print("split bn")
        try:
            net.set_bn_name("base_adv")
        except:
            net.model.set_bn_name("base_adv")

    # debug
    bx = torch.randn(2, 3, 32, 32).to(device)
    logits, feat_dict = net(bx, get_feat=True)
    print("logits:", logits.shape)
    print("len(feat_dict):", len(feat_dict))

    # print(net)

    if args.is_update_bn:
        print("update bn")
        from swa import moving_average, bn_update, bn_update_adv

        adversary = attacks.PGD_linf(epsilon=8.0 / 255, num_steps=10, step_size=2.0 / 255).cuda()
        bn_update_adv(train_loader, net, adversary)

    ############################################
    ############# Multi-GPU Setting ############
    ############################################

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net).to(device)

    ##################################
    ######## Eval Auto Attack ########
    ##################################
    _, test_acc = evaluate(args, net, test_loader, adv=False)
    print("test_acc:", test_acc)

    _, adv_acc = evaluate(args, net, test_loader)
    print("adv_acc:", adv_acc)

    exit()

    _, adv_acc_AA = evaluate_auto_attack(args, net, test_loader)
    print("adv_acc_AA:", adv_acc_AA)

    ##################################
    ######## LOG Results #############
    ##################################
    log_dir = os.path.join(args.log_dir, args.dataset, args.model, args.mark)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "eval.txt")
    with open(log_path, "w") as f:
        f.write(f"test_acc: {test_acc}\n")
        f.write(f"adv_acc: {adv_acc}\n")
        f.write(f"adv_acc_AA: {adv_acc_AA}\n")
