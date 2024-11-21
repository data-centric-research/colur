#!/bin/bash

# 检查是否传递了GPU的参数
if [ -z "$1" ]; then
    echo "使用方法: ./run_cifar_replay.sh <GPU_ID>"
    exit 1
fi

# 设置指定的 GPU
export CUDA_VISIBLE_DEVICES=$1
echo "Using GPU $1"
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python ./baseline_code/colearn/main_disc_cvpr.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --uni_name GJS --learning_rate 1e-3 --optimizer adam --batch_size 64 --noise_type asymmetric --noise_ratio 0.1
python ./baseline_code/colearn/main_disc_cvpr.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --uni_name GJS --learning_rate 1e-3 --optimizer adam --batch_size 64 --noise_type asymmetric --noise_ratio 0.25
python ./baseline_code/colearn/main_disc_cvpr.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --uni_name GJS --learning_rate 1e-3 --optimizer adam --batch_size 64 --noise_type asymmetric --noise_ratio 0.5
python ./baseline_code/colearn/main_disc_cvpr.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --uni_name GJS --learning_rate 1e-3 --optimizer adam --batch_size 64 --noise_type asymmetric --noise_ratio 0.75
python ./baseline_code/colearn/main_disc_cvpr.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --uni_name GJS --learning_rate 1e-3 --optimizer adam --batch_size 64 --noise_type asymmetric --noise_ratio 0.9