import glob
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as T
import torchvision.utils
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def get_dataloaders(args, shuffle_test=False):
    """
    Arguments:
        args:
            dataset: dataset name
            data_root: root directory
            batch_size: training batch size
            test_bs: test batch size
        shuffle_test:
            whether to shuffle test data
            
    Returns:
        train_loader: training data loader
        test_loader: test data loader
        num_classes: number of classes in the dataset
    """
    if args.dataset == "cifar10":
        train_transform = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
        test_transform = T.ToTensor()

        train_data = datasets.CIFAR10(
            root=args.data_root,
            train=True,
            transform=train_transform,
            download=True,
        )

        test_data = datasets.CIFAR10(
            root=args.data_root,
            train=False,
            transform=test_transform,
            download=True,
        )
        num_classes = 10

    elif args.dataset == "cifar100":
        train_transform = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
        test_transform = T.ToTensor()

        train_data = datasets.CIFAR100(
            root=args.data_root,
            train=True,
            transform=train_transform,
            download=True,
        )
        test_data = datasets.CIFAR100(
            root=args.data_root,
            train=False,
            transform=test_transform,
            download=True,
        )
        num_classes = 100

    elif args.dataset == "imagenette-160":
        img_size = 128
        train_transform = T.Compose(
            [
                T.RandomCrop(img_size, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )
        test_transform = T.Compose(
            [
                T.CenterCrop(img_size),
                T.ToTensor(),
            ]
        )

        root = os.path.join(args.data_root, "imagenette2-160")
        train_dir = os.path.join(root, "train")
        test_dir = os.path.join(root, "val")
        train_data = datasets.ImageFolder(train_dir, transform=train_transform)
        test_data = datasets.ImageFolder(test_dir, transform=test_transform)
        num_classes = 10

    elif args.dataset == "svhn":
        train_transform = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
        test_transform = T.ToTensor()

        train_data = datasets.SVHN(
            root=args.data_root,
            split="train",
            transform=train_transform,
            download=False,
        )
        test_data = datasets.SVHN(
            root=args.data_root,
            split="test",
            transform=test_transform,
            download=False,
        )
        num_classes = 10

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_bs,
        shuffle=shuffle_test,
        num_workers=8,
        pin_memory=True,
    )
    return train_loader, test_loader, num_classes
