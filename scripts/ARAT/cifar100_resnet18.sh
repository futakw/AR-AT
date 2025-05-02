#!/bin/sh

#SBATCH -J c100_res18
#SBATCH -p qgpu
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH -o ./slurm_logs/slurm-%j-%a.out

########### DATASET ###########
# dataset=cifar10
dataset=cifar100



########### MODEL ###########

# RESNET18

model=cifar_resnet18_split_bn
align_layers="layer4.0.bn1,layer4.0.bn2,layer4.1.bn1,layer4.1.bn2"
align_features_weight=30.0

# WRN-34-10

# model=cifar_wrn_34_10_split_bn
# align_layers="block3.layer.0.relu1,block3.layer.0.relu2,block3.layer.1.relu1,block3.layer.1.relu2,block3.layer.2.relu1,block3.layer.2.relu2,block3.layer.3.relu1,block3.layer.3.relu2,block3.layer.4.relu1,block3.layer.4.relu2,avgpool"
# align_features_weight=100.0



########### SEED ###########
seed=0


########### METHOD ###########

#### ARAT ###

mark=ARAT
is_auto_balance_ce_loss=True
is_use_predictor_xx=True
align_type="x->y"
bn_names="base,base_adv"
is_swa=False
swa_freq=-1
swa_start=-1

#### ARAT+SWA ###

# mark=ARAT+SWA
# is_auto_balance_ce_loss=True
# is_use_predictor_xx=True
# align_type="x->y"
# bn_names="base,base_adv"
# is_swa=True
# swa_freq=500
# swa_start=76




python3 train_ARAT.py \
    --dataset $dataset \
    --model $model \
    --align_features_weight $align_features_weight \
    --align_layers $align_layers \
    --is_use_predictor_xx $is_use_predictor_xx \
    --align_type $align_type \
    --mark $mark \
    --bn_names $bn_names \
    --is_auto_balance_ce_loss $is_auto_balance_ce_loss \
    --is_swa $is_swa \
    --swa_freq $swa_freq \
    --save_interval 100 \
    --eval_interval 100 \
    --swa_start $swa_start \
    --seed $seed