# -*- coding: utf-8 -*-
import argparse
import json
import os
import pandas as pd
import pprint
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import attacks
from eval import evaluate, evaluate_auto_attack
from loss import AlignLoss
from models.load_model import load_model
from utils.config_utils import str2bool as s2b, str2strlist as s2sl, str2intlist as s2il
from utils.data import get_dataloaders
from utils.utils import Tee
from utils.swa import moving_average, bn_update_adv


def get_args():
    parser = argparse.ArgumentParser(description="Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Seed and logging
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Folder to save logs.")
    parser.add_argument("--overwrite", type=s2b, default=False)
    parser.add_argument("--save_root_dir", type=str, default="./ckpts")
    parser.add_argument("--mark", type=str, default="ARAT")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=5)

    # Model & Dataset
    parser.add_argument("--model", type=str, default="cifar_resnet18")
    parser.add_argument("--bn_names", type=s2sl, default=["base", "base_adv"], help="BN layer names to split")
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--num_classes", type=int, default=10)

    # Optimization
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_bs", type=int, default=256)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=0.0005)

    # Attack
    parser.add_argument("--attack_norm", type=str, default="inf")
    parser.add_argument("--epsilon", type=float, default=8.0 / 255)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--step_size", type=float, default=2.0 / 255)

    # Eval attack
    parser.add_argument("--eval_epsilon", type=float, default=8.0 / 255)
    parser.add_argument("--eval_num_steps", type=int, default=20)
    parser.add_argument("--eval_step_size", type=float, default=2.0 / 255)
    parser.add_argument("--is_eval_auto_attack", type=s2b, default=True)
    parser.add_argument("--eval_auto_attack_ep", type=s2il, default=[])

    # Feature regularization
    parser.add_argument("--feat_align_loss_metric", type=str, default="cos-sim")
    parser.add_argument("--align_features_weight", type=float, default=30.0)
    parser.add_argument("--is_avg_pool", type=s2b, default=True, help="Apply avg_pool to feature map")
    parser.add_argument("--is_relu", type=s2b, default=True, help="Apply relu to feature map")
    parser.add_argument("--is_use_predictor_xx", type=s2b, default=True, help="Use predictor MLP head on adv. features")
    parser.add_argument("--is_use_predictor_x", type=s2b, default=False, help="Use predictor MLP head on clean features")
    parser.add_argument("--pred_dim_ratio", type=float, default=0.25, help="Predictor's hidden dim (rel.)")
    parser.add_argument("--align_type", type=str, default="x->y", help="Align type: x->y (stop-grad(y))")
    parser.add_argument("--align_layers", type=s2sl, default=[], help="Specify layer names to regularize")
    parser.add_argument("--align_layers_name", type=str, default=None, help="Just for logging")

    # Loss weights
    parser.add_argument("--ce_loss_fxx_y_weight", type=float, default=1.0, help="Weight for adv. classification loss")
    parser.add_argument("--ce_loss_gx_y_weight", type=float, default=1.0, help="Weight for clean classification loss")
    parser.add_argument("--is_auto_balance_ce_loss", type=s2b, default=True, help="Auto-balance classification loss")
    parser.add_argument("--auto_bal_prev_acc_weight", type=float, default=1.0, help="Weight for prev. accuracy in auto-balance")

    # SWA
    parser.add_argument("--is_swa", type=s2b, default=False)
    parser.add_argument("--swa_start", type=int, default=50)
    parser.add_argument("--swa_decay", type=str, default="n")
    parser.add_argument("--swa_freq", type=int, default=500)
    parser.add_argument("--swa_n", type=int, default=0)

    return parser.parse_args()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def save_checkpoint(model, path, optimizer=None, scheduler=None):
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    # remove featExtract wrapper
    model = model.model
    torch.save(model_to_save.state_dict(), path)
    if optimizer and scheduler:
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, path.replace(".pt", "_optim.pt"))
        

def load_checkpoint(model, path):
    model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
    checkpoint = torch.load(path)
    model_to_load.load_state_dict(checkpoint)
    if "optimizer" in checkpoint:
        optimizer = checkpoint["optimizer"]
        scheduler = checkpoint["scheduler"]
        return optimizer, scheduler
    return None, None


def train_one_epoch(args, epoch, net, train_loader, optimizer, adversary, feature_pair_loss_dict, prev_train_acc, device, bn_adv="adv", bn_clean="clean", net_swa=None):
    """
    Train one epoch with adversarial training.
    Args:
        net: model
        train_loader: training data loader
        optimizer: optimizer
        adversary: adversarial attack
        feature_pair_loss_dict: feature regularization loss functions (dict)
        prev_train_acc: previous training accuracy
        device: device (GPU or CPU)
        bn_adv: name of the adversarial batch normalization
        bn_clean: name of the clean batch normalization
    Returns:
        loss_total: average loss over the epoch
    """
    net.train()
    net.model.reset_bn_name()
    loss_total = 0.0
    for batch_idx, (bx, by) in tqdm(enumerate(train_loader), total=len(train_loader)):
        bx, by = bx.to(device), by.to(device)

        # Attack (using the main net w/ adv. bn)
        net.model.set_bn_name(bn_adv)
        adv_bx = adversary(net, bx, by) if args.attack_norm == "inf" else adversary(bx, by)
        net.model.reset_bn_name()

        # Forward
        logits_fx, feat_dict_fx = net(bx, get_feat=True, bn_name=bn_clean)
        logits_fxx, feat_dict_fxx = net(adv_bx, get_feat=True, bn_name=bn_adv)

        # Loss
        loss_dict = {}

        # (Loss 1) classification loss
        w_fxx = prev_train_acc ** args.auto_bal_prev_acc_weight if args.is_auto_balance_ce_loss else args.ce_loss_fxx_y_weight
        w_fx = (1 - prev_train_acc) ** args.auto_bal_prev_acc_weight if args.is_auto_balance_ce_loss else args.ce_loss_gx_y_weight

        loss_dict["ce_loss_fxx_y"] = F.cross_entropy(logits_fxx, by) * w_fxx
        loss_dict["ce_loss_fx_y"] = F.cross_entropy(logits_fx, by) * w_fx

        # (Loss 2) feature regularization loss
        for layer_name in args.align_layers:
            feat_fxx = feat_dict_fxx[layer_name]
            feat_fx = feat_dict_fx[layer_name]
            w = args.align_features_weight / len(args.align_layers)
            # Note: the order of feat matters for assymetric loss
            reg_loss = feature_pair_loss_dict[layer_name](feat_fxx, feat_fx) * w  
            loss_dict[f"align_feat_fxx_fx-{layer_name}"] = reg_loss

        loss = sum(loss_dict.values())
        if torch.isnan(loss):
            raise ValueError("Loss became NaN.")

        loss_total += float(loss.data)

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # SWA
        step = epoch * len(train_loader) + batch_idx
        if args.is_swa and args.swa_start <= epoch and step % args.swa_freq == 0:
            print("SWA update")
            if isinstance(args.swa_decay, str):
                moving_average(net_swa, net, 1.0 / (args.swa_n + 1))
                args.swa_n += 1
            else:
                moving_average(net_swa, net, args.swa_decay)
                
    return loss_total / len(train_loader)


def test_sep_bn(net, loader, bn_adv="adv", bn_clean="clean"):
    net.eval()
    loss_total = 0.0
    correct, correct_sub = 0, 0
    with torch.no_grad():
        for (bx, by) in loader:
            bx, by = bx.cuda(), by.cuda()
            logits = net(bx, bn_name=bn_adv)
            logits_sub = net(bx, bn_name=bn_clean)
            loss = F.cross_entropy(logits, by)
            # accuracy
            pred = logits.data.max(1)[1]
            correct += pred.eq(by.data).sum().item()
            pred_sub = logits_sub.data.max(1)[1]
            correct_sub += pred_sub.eq(by.data).sum().item()
            # test loss average
            loss_total += float(loss.data)
    n = len(loader.dataset)
    return loss_total / len(loader), correct / n, correct_sub / n


def test_single_bn(net, loader, max_n=1000, bn_name="clean"):
    net.eval()
    loss_avg = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for (bx, by) in loader:
            bx, by = bx.cuda(), by.cuda()
            logits = net(bx, bn_name=bn_name)
            loss = F.cross_entropy(logits, by)
            pred = logits.data.max(1)[1]
            correct += pred.eq(by.data).sum().item()
            loss_avg += float(loss.data)
            n += bx.size(0)
            if n >= max_n:
                break
    return loss_avg / n, correct / n


def main():
    args = get_args()
    setup_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.align_layers_name is None:
        args.align_layers_name = "-".join(args.align_layers)

    # Set Loggers
    SAVE_DIR = os.path.join(
        args.save_root_dir,
        args.dataset,
        args.model,
        args.mark,
        f"attack_norm={args.attack_norm}" if args.attack_norm != "inf" else "",
        f"seed{args.seed}" if args.seed != 1 else "",
    )

    SAVE_DIR = os.path.normpath(SAVE_DIR)
    is_done_path = os.path.join(SAVE_DIR, "done")
    if not args.overwrite and os.path.exists(is_done_path):
        print("Save directory already exists:", SAVE_DIR)
        exit()
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("SAVE_DIR: ", SAVE_DIR)

    # Writer
    writer_path = os.path.join(args.log_dir, SAVE_DIR)
    writer = SummaryWriter(writer_path)

    # df path
    df_path = os.path.join(SAVE_DIR, "results.csv")
    df_columns = ["epoch", "time(s)", "train_loss", "test_loss", "test_acc", "adv_acc", "test_acc_sub"]
    df = pd.DataFrame(columns=df_columns)

    # Save args with json
    save_json(os.path.join(SAVE_DIR, "args.json"), vars(args))

    # Logger
    sys.stdout = Tee(os.path.join(SAVE_DIR, "out.txt"))
    sys.stderr = Tee(os.path.join(SAVE_DIR, "err.txt"))


    # Data
    train_loader, test_loader, num_classes = get_dataloaders(args)
    args.num_classes = num_classes

    # Create Models
    net = load_model(args, args.model, num_classes, extract_layers=args.align_layers,
                       is_avg_pool=args.is_avg_pool, is_relu=args.is_relu).to(device)
    
    net_swa = None
    if args.is_swa:
        net_swa = load_model(args, args.model, num_classes, extract_layers=args.align_layers, is_avg_pool=args.is_avg_pool, is_relu=args.is_relu).to(device)


    # Feature Loss Setup
    net.model.set_bn_name(args.bn_names[0])
    with torch.no_grad():
        dims = {
            name: feat.shape[1]
            for name, feat in net(torch.randn(1, 3, 32, 32).to(device), get_feat=True)[1].items()
        }
    net.model.reset_bn_name()

    feature_pair_loss_dict = {
        name: AlignLoss(
            args,
            loss_metric=args.feat_align_loss_metric,
            is_use_predictor_x=args.is_use_predictor_xx,
            is_use_predictor_y=args.is_use_predictor_x,
            feat_dim=dims[name],
            hidden_dim=int(dims[name] * args.pred_dim_ratio),
            align_type=args.align_type,
        ).to(device)
        for name in args.align_layers
    }
    if args.is_use_predictor_xx or args.is_use_predictor_x:
        for name in args.align_layers:
            print(f"Create predictor for {name}: feat_dim={dims[name]}, hidden_dim={int(dims[name] * args.pred_dim_ratio)}")

    # Multi-GPU Setting 
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net).to(device)

    # Optimizer
    to_optimize_params = list(net.parameters())
    if args.is_use_predictor_xx or args.is_use_predictor_x:
        print("Add predictor parameters to optimizer. ")
        for k in feature_pair_loss_dict:
            to_optimize_params += list(feature_pair_loss_dict[k].parameters())

    optimizer = torch.optim.SGD(to_optimize_params, args.learning_rate, momentum=args.momentum, weight_decay=args.decay)

    # Scheduler
    def get_lr(epoch):
        """epoch starts from 1."""
        init_lr = args.learning_rate
        if epoch <= 75:
            lr = init_lr
        elif 76 <= epoch <= 90:
            lr = init_lr * 0.1
        elif 91 <= epoch <= 100:
            lr = init_lr * 0.01
        elif 101 <= epoch <= 110:
            lr = init_lr * 0.001
        elif epoch >= 111:
            lr = init_lr * 0.0001
        return lr

    class CustomLR(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, last_epoch=-1):
            """assuming that epoch starts from 1."""
            super(CustomLR, self).__init__(optimizer, last_epoch)

            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.02

        def step(self, epoch=None):
            if epoch is None:
                epoch = self.last_epoch
            else:
                self.last_epoch = epoch

            lr = get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            print("scheduler step(): epoch={}, lr={}".format(epoch, lr))

    scheduler = CustomLR(optimizer)

    # Adv. attack
    if args.attack_norm == "inf":
        adversary = attacks.PGD_linf(
            epsilon=args.epsilon, num_steps=args.num_steps, step_size=args.step_size
        ).cuda()
    elif args.attack_norm == "2":
        import torchattacks

        adversary = torchattacks.PGDL2(
            net, eps=args.epsilon, alpha=args.step_size, steps=args.num_steps, random_start=True
        )
    else:
        raise NotImplementedError

    # BN names
    if "adv" in args.bn_names:
        BN_ADV = "adv"
        BN_CLEAN = "clean"
    else:
        BN_ADV = "adv"
        BN_CLEAN = "clean"

    # Training
    print("----\nBeginning Training")

    start_epoch = 1
    adv_acc = -1
    for epoch in range(start_epoch, args.epochs + 1):
        begin_epoch = time.time()

        # Get previous training accuracy
        _, prev_train_acc = test_single_bn(net, train_loader, bn_name=BN_CLEAN)
        print("prev_train_acc", prev_train_acc)

        # train one epoch
        train_loss = train_one_epoch(
            args,
            epoch,
            net,
            train_loader,
            optimizer,
            adversary,
            feature_pair_loss_dict,
            prev_train_acc,
            device,
            bn_adv=BN_ADV,
            bn_clean=BN_CLEAN,
            net_swa=net_swa,
        )

        # test clean accuracy
        test_loss, test_acc, test_acc_sub = test_sep_bn(net, test_loader)

        # Scheduler
        scheduler.step(epoch)

        # Save model
        model_save_path = os.path.join(SAVE_DIR, f"epoch_{epoch}.pt")
        if epoch % args.save_interval == 0:
            # save model (remove featExtract wrapper)
            save_checkpoint(net, model_save_path, optimizer, scheduler)

            # save feature_pair_loss_dict (predictor MLP head)
            for k in feature_pair_loss_dict:
                torch.save(feature_pair_loss_dict[k], model_save_path.replace(".pt", f"_pred_{k}.pt"))

            if args.is_swa:
                torch.save(net_swa.state_dict(), model_save_path.replace(".pt", "_swa.pt"))

        # Evaluation
        if epoch % args.eval_interval == 0:
            net.model.set_bn_name(BN_ADV)
            _, adv_acc = evaluate(
                net,
                test_loader,
                lpnorm=args.attack_norm,
                eps=args.eval_epsilon,
                steps=args.eval_num_steps,
                alpha=args.eval_step_size,
            )
            net.model.reset_bn_name()
            print("adv_acc:", adv_acc)

            if args.is_swa and epoch > args.swa_start:
                print("eval swa")
                net_swa.model.set_bn_name(BN_ADV)
                print("update bn")
                bn_update_adv(train_loader, net_swa, adversary)

                _, adv_acc_swa = evaluate(
                    net_swa,
                    test_loader,
                    lpnorm=args.attack_norm,
                    eps=args.eval_epsilon,
                    steps=args.eval_num_steps,
                    alpha=args.eval_step_size,
                )
                net_swa.model.reset_bn_name()
                writer.add_scalar("adv_acc_swa", adv_acc_swa, epoch)
                print("adv_acc_swa:", adv_acc_swa)

                # eval clean acc
                net_swa.model.set_bn_name(BN_ADV)
                _, test_acc_swa = test_single_bn(net_swa, test_loader, max_n=100000, bn_name=BN_ADV)
                net_swa.model.reset_bn_name()
                print("clean acc swa: ", test_acc_swa)

                if args.is_swa:
                    model_save_path = os.path.join(SAVE_DIR, f"last_swa.pt")
                    torch.save(net_swa.state_dict(), model_save_path)
        else:
            adv_acc = -1

        if epoch in args.eval_auto_attack_ep:
            net.model.set_bn_name(BN_ADV)
            _, adv_acc_AA = evaluate_auto_attack(net, test_loader, args.num_classes, lpnorm=args.attack_norm)
            writer.add_scalar("adv_acc_AA", adv_acc_AA, epoch)
            print("adv_acc_AA:", adv_acc_AA)

        # Logging
        # save to csv
        df_new_row = {
            "epoch": epoch,
            "time(s)": int(time.time() - begin_epoch),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_acc": test_acc * 100.0,
            "adv_acc": adv_acc * 100.0,
            "test_acc_sub": test_acc_sub * 100.0,
        }
        df = pd.concat([df, pd.DataFrame([df_new_row])], ignore_index=True)
        df.to_csv(df_path, index=False)
        pprint.pprint(df_new_row)

        # save to tensorboard
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("test_loss", test_loss, epoch)
        writer.add_scalar("test_acc", test_acc, epoch)
        if adv_acc > 0:
            writer.add_scalar("adv_acc", adv_acc, epoch)
        writer.add_scalar("test_acc_sub", test_acc_sub, epoch)


    # Save Last Model
    #  (remove featExtract wrapper)
    model_save_path = os.path.join(SAVE_DIR, "last.pt")
    save_checkpoint(net, model_save_path, optimizer, scheduler)


    # Eval Auto Attack
    if args.is_eval_auto_attack:
        net.model.set_bn_name(BN_ADV)
        _, adv_acc_AA = evaluate_auto_attack(net, test_loader, args.num_classes, lpnorm=args.attack_norm)
        writer.add_scalar("adv_acc_AA", adv_acc_AA, epoch)
        print("adv_acc_AA:", adv_acc_AA)

        if args.is_swa:
            print("eval swa")
            net_swa.model.set_bn_name(BN_ADV)
            bn_update_adv(train_loader, net_swa, adversary)
            _, adv_acc_AA_swa = evaluate_auto_attack(net_swa, test_loader, args.num_classes, lpnorm=args.attack_norm)
            writer.add_scalar("adv_acc_AA_swa", adv_acc_AA_swa, epoch)
            print("adv_acc_AA_swa:", adv_acc_AA_swa)

            # eval clean acc
            net_swa.model.set_bn_name(BN_ADV)
            _, test_acc_swa = test_single_bn(net_swa, test_loader, max_n=100000, bn_name=BN_ADV)
            net_swa.model.reset_bn_name()
            print("clean acc swa: ", test_acc_swa)
    else:
        adv_acc_AA = -1


    # Save Final Metrics
    metrics = {
        "test_acc": test_acc,
        "adv_acc": adv_acc,
        "adv_acc_AA": adv_acc_AA,
        "test_acc_sub": test_acc_sub,
    }

    args_dict = vars(args)
    for k, v in args_dict.items():
        if type(v) == list:
            args_dict[k] = str(v)
    writer.add_hparams(args_dict, metrics)
    writer.close()

    d = {"args": args_dict, "metrics": metrics}
    save_json(os.path.join(SAVE_DIR, "hparams_metrics.json"), d)

    ed = time.time()
    print("Total time: {} / min".format((ed - st) / 60))

    # done
    with open(is_done_path, "a") as f:
        f.write("done")
    print("\nDone")


if __name__ == "__main__":
    st = time.time()
    main()
    print("Total time: {} / min".format((time.time() - st) / 60))
    print("Done")