import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

criterion_kl = nn.KLDivLoss(size_average=True)
cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)


def normalize_feat(feat):
    assert len(feat.shape) == 2
    return F.normalize(feat, p=2, dim=1)


def logit_pair_loss(logits_input, logits_target, T=1.0):
    loss = criterion_kl(
        F.log_softmax(logits_input / T, dim=1),
        F.softmax(logits_target / T, dim=1),
    )
    return loss


class AlignLoss(nn.Module):
    """
    Regularization loss to align two latent representations.
    For ARAT, we use this loss to align the main model and the sub model, which shares all the layers except the BN layers.
    By default, ARAT takes asymmetric alignment, i.e., only aligning the main model to the sub model (x->y).

    Args:
        args: arguments
        loss_metric: loss function, including "cos-sim", "mse", "kl".
            By default, we use "cos-sim".
        align_type: alignment type, including "x->y", "y->x", "x->y,y->x", "x<->y".
            By default, we use "x->y".
        is_use_predictor: whether to use predictor
            By default, we use predictor, following SimSiam.
        feat_dim: feature dimension
        hidden_dim: dimension of bottleneck structure in predictor
            By default, we set it to 1/4 of the feature dimension.
    """

    def __init__(
        self,
        args,
        loss_metric="cos-sim",
        align_type="x->y",
        # is_use_predictor=True,
        is_use_predictor_x=False,
        is_use_predictor_y=False,
        is_ise_shared_predictor=False,
        predictor_version="v1",
        feat_dim=512,
        hidden_dim=512,
    ):
        """
        prev_dim: feature dimension
        dim: projection dimension
        pred_dim: dimension of bottleneck structure in predictor
        """
        super().__init__()
        self.args = args
        self.loss_metric = loss_metric
        self.align_type = align_type
        self.is_use_predictor_x = is_use_predictor_x
        self.is_use_predictor_y = is_use_predictor_y
        assert align_type in ["x->y", "y->x", "x->y,y->x", "x<->y"]

        # projection
        if is_use_predictor_x or is_use_predictor_y:
            self.feat_dim = feat_dim
            self.hidden_dim = hidden_dim

            if predictor_version == "v1":
                # build a 2-layer predictor.
                # Based on SimSiam:
                # https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py
                def make_predictor():
                    return nn.Sequential(
                        nn.Linear(feat_dim, hidden_dim, bias=False),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(inplace=True),  # hidden layer
                        nn.Linear(hidden_dim, feat_dim),
                    )
            elif predictor_version == "v2":
                pass

            if is_use_predictor_x:
                self.predictor_x = make_predictor()
            if is_use_predictor_y:
                if is_ise_shared_predictor:
                    self.predictor_y = self.predictor_x
                else:
                    self.predictor_y = make_predictor()


    def loss_func(self, x, y, **kwargs):
        if self.loss_metric == "cos-sim":
            return 1 - F.cosine_similarity(x, y, dim=-1).mean()

        elif self.loss_metric == "mse":
            return F.mse_loss(x, y)
        
        elif self.loss_metric == "kl":
            T = self.args.kd_temp
            loss = (
                F.kl_div(
                    F.log_softmax(x / T, dim=1),
                    F.softmax(y / T, dim=1),
                    reduction="batchmean",
                )
                * T
                * T
            )
            return loss


    def forward(self, x, y):
        """
        x: features from main model
        y: features from sub model
        """
        assert x.shape == y.shape, f"x.shape: {x.shape}, y.shape: {y.shape}"
        # if self.is_use_predictor:
        #     # if len(x.shape) == 4:
        #     #     x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        #     #     y = F.avg_pool2d(y, y.size()[2:]).view(y.size(0), -1)

        #     if self.align_type in ["x->y", "x->y,y->x", "x<->y"]:
        #         px = self.predictor(x)
        #     if self.align_type in ["y->x", "x->y,y->x", "x<->y"]:
        #         py = self.predictor(y)

        kwargs = {}
        if self.align_type == "x->y":
            if self.is_use_predictor_x and self.is_use_predictor_y:
                px = self.predictor_x(x)
                y = y.detach()
                py = self.predictor_y(y)
                return self.loss_func(px, py, **kwargs)
            elif self.is_use_predictor_x:
                px = self.predictor_x(x)
                return self.loss_func(px, y.detach(), **kwargs)
            elif self.is_use_predictor_y:
                y = y.detach()
                py = self.predictor_y(y)
                return self.loss_func(x, py, **kwargs)
            else:
                return self.loss_func(x, y.detach(), **kwargs)
        elif self.align_type == "y->x":
            if self.is_use_predictor_x and self.is_use_predictor_y:
                py = self.predictor_y(y)
                x = x.detach()
                px = self.predictor_x(x)
                return self.loss_func(py, px, **kwargs)
            elif self.is_use_predictor_x:
                px = self.predictor_x(x)
                return self.loss_func(y, px, **kwargs)
            elif self.is_use_predictor_y:
                py = self.predictor_y(y)
                return self.loss_func(x, py, **kwargs)
            else:
                return self.loss_func(y, x.detach(), **kwargs)
        elif self.align_type == "x->y,y->x":
            if self.is_use_predictor_x and self.is_use_predictor_y:
                px = self.predictor(x)
                py = self.predictor(y)
                loss = 0
                loss += self.loss_func(px, y.detach())
                loss += self.loss_func(py, x.detach())
                return loss / 2
            elif self.is_use_predictor_x:
                px = self.predictor_x(x)
                loss = 0
                loss += self.loss_func(px, y.detach())
                loss += self.loss_func(y, x.detach())
                return loss / 2
            elif self.is_use_predictor_y:
                py = self.predictor_y(y)
                loss = 0
                loss += self.loss_func(py, x.detach())
                loss += self.loss_func(x, y.detach())
                return loss / 2
            else:
                loss = 0
                loss += self.loss_func(x, y.detach())
                loss += self.loss_func(y, x.detach())
                return loss / 2
        elif self.align_type == "x<->y":
            return self.loss_func(x, y, **kwargs)


def get_pair_loss_func(args, metric, **kwargs):
    if metric == "cos-sim":

        def neg_cos_sim(x, y):
            return 1 - F.cosine_similarity(x, y, dim=-1).mean()

        return neg_cos_sim

    elif metric == "mse":

        def mse(x, y):
            return F.mse_loss(x, y)

        return mse

    elif metric == "kl":

        def kl_divergence(x, y):
            T = args.kd_temp
            loss = (
                F.kl_div(
                    F.log_softmax(x / T, dim=1),
                    F.softmax(y / T, dim=1),
                    reduction="batchmean",
                )
                * T
                * T
            )
            return loss

        return kl_divergence

    else:
        raise NotImplementedError
