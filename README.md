# Official codes for Asymmetrically Representation Regularized Adversarial Training (ARAT) [ICLR'25]
### Author: [Futa Waseda](https://futa-waseda.netlify.app/)
### Paper: [Rethinking Invariance Regularization in Adversarial Training to Improve Robustness-Accuracy Trade-off](https://openreview.net/forum?id=H1g0v4r9t7)

## Overview
This repository contains the official implementation of ARAT, which is a method to improve the robustness-accuracy trade-off in adversarial training. The code is based on the [PyTorch](https://pytorch.org/) framework and is designed to be easy to use and modify.

## Method
<!-- pdf -->
![ARAT](https://raw.githubusercontent.com/futawaseda/AR-AT/assets/method_fig.pdf)

## Results
<!-- pdf -->
![ARAT](https://raw.githubusercontent.com/futawaseda/AR-AT/assets/results_fig.pdf)

## Quick try
- Environment setup
```
pip install -r requirements.txt
```

- Train ARAT: ResNet-18 on CIFAR10.
```
bash scripts/cifar10_resnet18.sh
```
- Train ARAT: WRN-34-10 on CIFAR10.
```
bash scripts/cifar10_wrn-34-10.sh
```

