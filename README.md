# [ICLR'25] Official codes for Asymmetrically Representation Regularized Adversarial Training (ARAT)
- Author: [Futa Waseda](https://futa-waseda.netlify.app/)
- Paper: [Rethinking Invariance Regularization in Adversarial Training to Improve Robustness-Accuracy Trade-off [ICLR'25]](https://arxiv.org/abs/2402.14648)

## Method
<!-- pdf -->
![ARAT](https://github.com/futakw/AR-AT/blob/main/assets/method_fig.png)

## Results
<!-- pdf -->
![ARAT](https://github.com/futakw/AR-AT/blob/main/assets/result_fig.png)

## Quick try
### 1. Environment setup
```
conda create -n arat python=3.8
conda activate arat
conda install git pip -y
```

```
pip install -r requirements.txt
```

### 2-1. Train ARAT
CIFAR10, ResNet-18
```
bash scripts/ARAT/cifar10_resnet18.sh
```
CIFAR10, WideResNet-34-10
```
bash scripts/ARAT/cifar10_wrn-34-10.sh
```
CIFAR100, ResNet-18
```
bash scripts/ARAT/cifar100_resnet18.sh
```
CIFAR100, WideResNet-34-10
```
bash scripts/ARAT/cifar100_wrn-34-10.sh
```

### 2-2. Train ARAT+SWA
CIFAR10, ResNet-18
```
bash scripts/ARAT+SWA/cifar10_resnet18.sh
```
CIFAR10, WideResNet-34-10
```
bash scripts/ARAT+SWA/cifar10_wrn-34-10.sh
```
CIFAR100, ResNet-18
```
bash scripts/ARAT+SWA/cifar100_resnet18.sh
```
CIFAR100, WideResNet-34-10
```
bash scripts/ARAT+SWA/cifar100_wrn-34-10.sh
```

### 3. Model Weights
https://drive.google.com/file/d/1MmxURwmELuQw1dctmfvilPpejdBhxukO/view?usp=sharing

# Citation
```
@article{waseda2024rethinking,
  title={Rethinking Invariance Regularization in Adversarial Training to Improve Robustness-Accuracy Trade-off},
  author={Waseda, Futa and Chang, Ching-Chun and Echizen, Isao},
  journal={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
