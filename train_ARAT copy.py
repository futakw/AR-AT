# -*- coding: utf-8 -*-
import argparse
import json
import os
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
from utils.share_layers import share_layers
from utils.utils import Tee
from utils.swa import moving_average, bn_update_adv


def get_args():
    parser = argparse.ArgumentParser(description="Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Seed and logging
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Folder to save logs.")
    parser.add_argument("--overwrite", type=s2b, default=False)
    parser.add_argument("--is_debug", type=s2b, default=False)
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



args = get_args()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

setup_seed(args.seed)


############################################
################ Set Args ##################
############################################
if args.align_layers_name is None:
    args.align_layers_name = "-".join(args.align_layers)


############################################
################ Set Loggers ###############
############################################
SAVE_NAME = args.mark
SAVE_DIR = os.path.join(args.save_root_dir, args.dataset, args.model, SAVE_NAME)
if args.attack_norm != "inf":
    SAVE_DIR = os.path.join(SAVE_DIR, f"attack_norm={args.attack_norm}")
if args.seed != 1:
    SAVE_DIR = os.path.join(SAVE_DIR, f"seed{args.seed}")

is_done_path = os.path.join(SAVE_DIR, "done")
if not args.overwrite and os.path.exists(is_done_path):
    print("Save directory already exists:", SAVE_DIR)
    exit()
os.makedirs(SAVE_DIR, exist_ok=True)

# writer
writer_path = os.path.join(args.log_dir, SAVE_DIR)
writer = SummaryWriter(writer_path)

# df path
df_path = os.path.join(SAVE_DIR, "results.csv")
if not os.path.exists(df_path):
    columns = [
        "epoch",
        "time(s)",
        "train_loss",
        "test_loss",
        "test_acc",
        "adv_acc",
        "test_acc_sub",
    ]
    with open(df_path, "w") as f:
        f.write(",".join(columns) + "\n")

# Save args with json
with open(os.path.join(SAVE_DIR, "args.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

# logger
sys.stdout = Tee(os.path.join(SAVE_DIR, "out.txt"))
sys.stderr = Tee(os.path.join(SAVE_DIR, "err.txt"))

print("SAVE_NAME: ", SAVE_NAME)
print("SAVE_DIR: ", SAVE_DIR)
state = {k: v for k, v in args._get_kwargs()}
print(state)


############################################
################### Data ###################
############################################
train_loader, test_loader, num_classes = get_dataloaders(args)
args.num_classes = num_classes

############################################
############# Create Models ################
############################################
net = load_model(
    args,
    args.model,
    num_classes,
    extract_layers=args.align_layers,
    is_avg_pool=args.is_avg_pool,
    is_relu=args.is_relu,
)
net = net.to(device)

if args.is_swa:
    net_swa = load_model(
        args,
        args.model,
        num_classes,
        extract_layers=args.align_layers,
        is_avg_pool=args.is_avg_pool,
        is_relu=args.is_relu,
    )
    net_swa = net_swa.to(device)


############################################
###### Regularization Loss function ########
############################################
# feature regularization loss
#  Default:
#   - align all ReLU outputs in the last "block" of network.
#   - align features with cosine similarity.
#   - use predictor MLP head (h), with hidden dim being 1/4 of feature dim.
#   - align_type: x->y (stop-grad(y))
feature_pair_loss_dict = {}
# get dimensions of features
net.model.set_bn_name(args.bn_names[0])
with torch.no_grad():
    x = torch.randn(1, 3, 32, 32).to(device)
    _, feat_dict = net(x, get_feat=True)
    dims = {name: feat.shape[1] for name, feat in feat_dict.items()}
    print("dims: ", dims)
net.model.reset_bn_name()
# create loss functions for each layer. Predictor h is created inside, which has parameters to be optimized.
for align_layers_name in args.align_layers:
    feat_dim = dims[align_layers_name]
    hidden_dim = int(feat_dim * args.pred_dim_ratio)
    if args.is_use_predictor_xx or args.is_use_predictor_x:
        print("Create predictor for feature regularization loss.")
        print("{} -> feat_dim: {}, hidden_dim: {}".format(align_layers_name, feat_dim, hidden_dim))

    _feature_pair_loss = AlignLoss(
        args,
        loss_metric=args.feat_align_loss_metric,
        is_use_predictor_x=args.is_use_predictor_xx,  # use predictor on adv. features
        is_use_predictor_y=args.is_use_predictor_x,  # use predictor on clean features
        feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        align_type=args.align_type,
    ).to(device)
    feature_pair_loss_dict[align_layers_name] = _feature_pair_loss

############################################
############# Multi-GPU Setting ############
############################################
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = torch.nn.DataParallel(net).to(device)
    if args.awp:
        proxy = torch.nn.DataParallel(proxy).to(device)

############################################
################ Optimizer #################
############################################
to_optimize_params = list(net.parameters())

if args.is_use_predictor_xx or args.is_use_predictor_x:
    print("Add predictor parameters to optimizer. ")
    for k in feature_pair_loss_dict:
        to_optimize_params += list(feature_pair_loss_dict[k].parameters())

# optimizer
optimizer = torch.optim.SGD(
    to_optimize_params,
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.decay,
)


############################################
################# Scheduler ################
############################################
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

############################################
############### Adv. attack ################
############################################
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


############################################
############# Training Function ############
############################################
# BN names
if "base_adv" in args.bn_names:
    BN_BASE_ADV = "base_adv"
    BN_BASE = "base"
else:
    # same bn layers
    BN_BASE_ADV = "base"
    BN_BASE = "base"


##############################
############# TRAIN ############
##############################
def train(epoch, args=args, prev_train_acc=None):
    train_info_dict = {}

    net.train()
    net.model.reset_bn_name()


    timestamp = time.time()
    time_per_batch = []

    loss_total = 0.0
    sample_num = 0
    for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        bx, by = data
        bx, by = bx.cuda(), by.cuda()

        sample_num += bx.size(0)

        step = epoch * len(train_loader) + batch_idx

        loss_dict = {}
        
        #########################
        ######## attack #########
        #########################
        net.model.set_bn_name(BN_BASE_ADV)
        # print("bn_name: ", net.model.bn_name)
        if args.attack_norm == "inf":
            adv_bx = adversary(net, bx, by)
        elif args.attack_norm == "2":
            adv_bx = adversary(bx, by)
        else:
            raise NotImplementedError

        net.model.reset_bn_name()

        #########################
        ######## forward ########
        #########################
        # Get adv outputs from main net & clean outputs from sub net
        adv_output_dict = {}
        clean_output_dict = {}

        logits_fx, feat_dict_fx = net(bx, get_feat=True, bn_name=BN_BASE)
        clean_outputs = (logits_fx, feat_dict_fx)

        logits_fxx, feat_dict_fxx = net(adv_bx, get_feat=True, bn_name=BN_BASE_ADV)
        adv_outputs = (logits_fxx, feat_dict_fxx)

        ######################
        ######## loss ########
        ######################
        # classification loss

        # adv.
        # Auto-balance: the more (clean) accurate already, the more weight
        w = (
            prev_train_acc**args.auto_bal_prev_acc_weight
            if args.is_auto_balance_ce_loss
            else args.ce_loss_fxx_y_weight
        )
        loss_dict[f"ce_loss_fxx_y"] = F.cross_entropy(logits_fxx, by) * w

        # clean
        # Auto-balance: the more (clean) accurate already, the more weight
        w = (
            1 - prev_train_acc**args.auto_bal_prev_acc_weight
            if args.is_auto_balance_ce_loss
            else args.ce_loss_gx_y_weight
        )
        loss_dict[f"ce_loss_fx_y"] = F.cross_entropy(logits_fx, by) * w

        # feature regularization loss
        for i, layer_name in enumerate(args.align_layers):
            feat_fxx = feat_dict_fxx[layer_name]
            feat_fx = feat_dict_fx[layer_name]

            w = args.align_features_weight / len(args.align_layers)
            reg_loss = feature_pair_loss_dict[layer_name](feat_fxx, feat_fx) * w  # Note: the order of feat matters for assymetric loss
            loss_dict[f"feat_loss_fxx_fx-{layer_name}"] = reg_loss


        loss = sum(loss_dict.values())
        if torch.isnan(loss):
            raise ValueError("Loss became NaN.")

        loss_total += float(loss.data)


        #########################
        ######## update ########
        #########################
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
                

        if args.is_debug:
            break

        time_per_batch.append(time.time() - timestamp)
        timestamp = time.time()

    # print("time_per_batch: ", time_per_batch)
    print("time per batch: {} +- {}".format(np.mean(time_per_batch), np.std(time_per_batch)))

    state["train_loss"] = loss_total / len(train_loader)
    train_info_dict["train_loss"] = loss_total / len(train_loader)
    return train_info_dict


############################################
############### Test Function ##############
############################################
def test(loader):
    net.eval()

    loss_total = 0.0
    correct = 0
    correct_sub = 0
    with torch.no_grad():
        for data in loader:
            bx, by = data
            bx, by = bx.cuda(), by.cuda()

            with torch.cuda.amp.autocast():
                logits = net(bx, bn_name=BN_BASE_ADV)
                logits_sub = net(bx, bn_name=BN_BASE)

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


def test_single(loader, net, max_n=1000, bn_name=BN_BASE):
    net.eval()
    loss_avg = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for data in loader:
            bx, by = data
            bx, by = bx.cuda(), by.cuda()

            logits = net(bx, bn_name=bn_name)
            loss = F.cross_entropy(logits, by)

            # accuracy
            pred = logits.data.max(1)[1]
            correct += pred.eq(by.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

            n += bx.size(0)
            if n >= max_n:
                break
    return loss_avg / n, correct / n


def save_torch_model(net, path):
    model = net
    # if data parallel
    if isinstance(net, torch.nn.DataParallel):
        model = net.module
    # remove featExtract wrapper
    model = model.model
    torch.save(model.state_dict(), path)


############################################
############## Training Loop ###############
############################################
st = time.time()
print("Beginning Training\n")

start_epoch = 1
adv_acc = -1
train_info_dict_list = []
pre_purturbations = None
pre_feat_dict = None
pre_proj_feat_dict = None
pre_logits = None
pre_logits_std = None
pre_logits_from_proj_feat = None
for epoch in range(start_epoch, args.epochs + 1):
    state["epoch"] = epoch

    begin_epoch = time.time()

    ###################
    #### Train #####
    ###################
    _, prev_train_acc = test_single(train_loader, net)
    print("prev_train_acc", prev_train_acc)

    train_info_dict = train(epoch, prev_train_acc=prev_train_acc)
    train_info_dict_list.append(train_info_dict)

    train_loss = train_info_dict["train_loss"]

    test_loss, test_acc, test_acc_sub = test(test_loader)
    state["test_loss"] = test_loss
    state["test_accuracy"] = test_acc
    state["test_acc_sub"] = test_acc_sub


    ###################
    #### Scheduler ####
    ###################
    scheduler.step(epoch)

    ###################
    #### Save model ###
    ###################
    model_save_path = os.path.join(SAVE_DIR, f"epoch_{epoch}.pt")
    optim_save_path = os.path.join(SAVE_DIR, f"optim_{epoch}.pt")
    if epoch % args.save_interval == 0:
        # save model (remove featExtract wrapper)
        save_torch_model(net, model_save_path)

        optims = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(optims, optim_save_path)  # remove norm wrapper

        # save feature_pair_loss_dict (predictor MLP head)
        for k in feature_pair_loss_dict:
            torch.save(feature_pair_loss_dict[k], model_save_path.replace(".pt", f"_pred_{k}.pt"))

        if args.is_swa:
            torch.save(net_swa.state_dict(), model_save_path.replace(".pt", "_swa.pt"))

    ###################
    #### Evaluation ###
    ###################
    if epoch % args.eval_interval == 0:
        net.model.set_bn_name(BN_BASE_ADV)
        _, adv_acc = evaluate(
            args,
            net,
            test_loader,
            lpnorm=args.attack_norm,
            eps=args.eval_epsilon,
            steps=args.eval_num_steps,
            alpha=args.eval_step_size,
        )
        net.model.reset_bn_name()
        print("adv_acc:", adv_acc)

        if epoch % 5 == 0:
            net.model.set_bn_name(BN_BASE)
            _, adv_acc_sub = evaluate(
                args,
                net,
                test_loader,
                lpnorm=args.attack_norm,
                eps=args.eval_epsilon,
                steps=args.eval_num_steps,
                alpha=args.eval_step_size,
            )
            net.model.reset_bn_name()
            writer.add_scalar("adv_acc_sub", adv_acc_sub, epoch)
            print("adv_acc_sub:", adv_acc_sub)


        if args.is_swa and epoch > args.swa_start:
            print("eval swa")
            net_swa.model.set_bn_name(BN_BASE_ADV)
            print("update bn")
            bn_update_adv(train_loader, net_swa, adversary)

            _, adv_acc_swa = evaluate(
                args,
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
            net_swa.model.set_bn_name(BN_BASE_ADV)
            _, test_acc_swa = test_single(test_loader, net_swa, max_n=100000, bn_name=BN_BASE_ADV)
            net_swa.model.reset_bn_name()
            print("clean acc swa: ", test_acc_swa)

            if args.is_swa:
                model_save_path = os.path.join(SAVE_DIR, f"last_swa.pt")
                torch.save(net_swa.state_dict(), model_save_path)
    else:
        adv_acc = -1

    if epoch in args.eval_auto_attack_ep:
        net.model.set_bn_name(BN_BASE_ADV)
        _, adv_acc_AA = evaluate_auto_attack(args, net, test_loader, lpnorm=args.attack_norm)
        writer.add_scalar("adv_acc_AA", adv_acc_AA, epoch)
        print("adv_acc_AA:", adv_acc_AA)

    ###################
    #### Logging ######
    ###################
    log_data = (
        epoch,
        int(time.time() - begin_epoch),
        state["train_loss"],
        state["test_loss"],
        state["test_accuracy"] * 100.0,
        adv_acc * 100.0,
        state["test_acc_sub"] * 100.0,
        adv_acc_train * 100.0,
    )
    with open(df_path, "a") as f:
        f.write(
            "%03d,%05d,%0.6f,%0.5f,%0.2f,%0.2f,%0.2f,%0.2f\n"
            % (
                log_data[0],
                log_data[1],
                log_data[2],
                log_data[3],
                log_data[4],
                log_data[5],
                log_data[6],
                log_data[7],
            )
        )
    print(
        "\nEpoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Acc {4:.2f} | Adv Acc {5:.2f} | Test Acc Sub {6:.2f} | Train Adv Acc {7:.2f}\n".format(
            log_data[0],
            log_data[1],
            log_data[2],
            log_data[3],
            log_data[4],
            log_data[5],
            log_data[6],
            log_data[7],
        )
    )

    writer.add_scalar("test_acc", test_acc, epoch)
    if adv_acc > 0:
        writer.add_scalar("adv_acc", adv_acc, epoch)
    writer.add_scalar("test_acc_sub", test_acc_sub, epoch)


##################################
######## Save Last Model #########
##################################
# Save model (remove featExtract wrapper)
model_save_path = os.path.join(SAVE_DIR, "model_last.pt")
optim_save_path = os.path.join(SAVE_DIR, "optim_last.pt")
save_torch_model(net, model_save_path)

optims = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
torch.save(optims, optim_save_path)


##################################
######## Eval Auto Attack ########
##################################
if args.is_eval_auto_attack:
    net.model.set_bn_name(BN_BASE_ADV)
    _, adv_acc_AA = evaluate_auto_attack(args, net, test_loader, lpnorm=args.attack_norm)
    writer.add_scalar("adv_acc_AA", adv_acc_AA, epoch)
    print("adv_acc_AA:", adv_acc_AA)

    if args.is_swa:
        print("eval swa")
        net_swa.model.set_bn_name(BN_BASE_ADV)
        bn_update_adv(train_loader, net_swa, adversary)
        _, adv_acc_AA_swa = evaluate_auto_attack(args, net_swa, test_loader, lpnorm=args.attack_norm)
        writer.add_scalar("adv_acc_AA_swa", adv_acc_AA_swa, epoch)
        print("adv_acc_AA_swa:", adv_acc_AA_swa)

        # eval clean acc
        net_swa.model.set_bn_name(BN_BASE_ADV)
        _, test_acc_swa = test_single(test_loader, net_swa, max_n=100000, bn_name=BN_BASE_ADV)
        net_swa.model.reset_bn_name()
        print("clean acc swa: ", test_acc_swa)

else:
    adv_acc_AA = -1


##################################
##### Save Final Metrics #########
##################################
metrics = {
    "test_acc": test_acc,
    "adv_acc": adv_acc,
    "adv_acc_AA": adv_acc_AA,
    "test_acc_sub": test_acc_sub,
    "adv_acc_sub": adv_acc_sub,
}

args_dict = vars(args)
for k, v in args_dict.items():
    if type(v) == list:
        args_dict[k] = str(v)
print(args_dict)
writer.add_hparams(
    args_dict,
    metrics,
)
writer.close()

d = {
    "args": args_dict,
    "metrics": metrics,
}

with open(os.path.join(SAVE_DIR, "hparams_metrics.json"), "w") as f:
    json.dump(d, f, indent=4)

ed = time.time()
print("Total time: {} / min".format((ed - st) / 60))

# done
with open(is_done_path, "a") as f:
    f.write("done")
print("\nDone")
