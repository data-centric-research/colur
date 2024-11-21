#!/bin/bash

# 检查是否传递了GPU的参数
if [ -z "$1" ]; then
    echo "使用方法: ./this_script.sh <GPU_ID>"
    exit 1
fi

# 设置指定的 GPU
export CUDA_VISIBLE_DEVICES=$1
echo "Using GPU $1"
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python ./baseline_code/lips-mu/main_mu_cvpr.py --model efficientnet_s --dataset cifar-10 --num_epochs 10 --batch_size 256 --uni_name FT --unlearn_lr 1e-3 --noise_ratio 0.1 --alpha 1e-3
python ./baseline_code/lips-mu/main_mu_cvpr.py --model efficientnet_s --dataset cifar-10 --num_epochs 10 --batch_size 256 --uni_name FT --unlearn_lr 1e-3 --noise_ratio 0.25 --alpha 1e-3
python ./baseline_code/lips-mu/main_mu_cvpr.py --model efficientnet_s --dataset cifar-10 --num_epochs 10 --batch_size 256 --uni_name FT --unlearn_lr 1e-3 --noise_ratio 0.5 --alpha 1e-3
python ./baseline_code/lips-mu/main_mu_cvpr.py --model efficientnet_s --dataset cifar-10 --num_epochs 10 --batch_size 256 --uni_name FT --unlearn_lr 1e-3 --noise_ratio 0.75 --alpha 1e-3
python ./baseline_code/lips-mu/main_mu_cvpr.py --model efficientnet_s --dataset cifar-10 --num_epochs 10 --batch_size 256 --uni_name FT --unlearn_lr 1e-3 --noise_ratio 0.9 --alpha 1e-3