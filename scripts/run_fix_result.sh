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

# crul-flower-10
python ./core_model/CRUL.py --model wideresnet50 --dataset flower-102 --num_epochs 3 --repair_iter_num 5 --uni_name CRUL --learning_rate 1e-4 --optimizer adam --batch_size 32 --noise_type symmetric --noise_ratio 0.1
# crul-pet-10
python ./core_model/CRUL.py --model wideresnet50 --dataset pet-37 --num_epochs 3 --repair_iter_num 5 --uni_name CRUL --learning_rate 2e-5 --optimizer adam --batch_size 64 --noise_type asymmetric --noise_ratio 0.1 --mixup_alpha 0.5

# coteachingplus-pet-10%
python ./baseline_code/colearn/main_cvpr.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --uni_name Coteaching --learning_rate 1e-4 --optimizer adam --batch_size 64 --noise_type asymmetric --noise_ratio 0.1
# decoup-pet-10
python ./baseline_code/colearn/main_cvpr.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --uni_name Decoupling --learning_rate 1e-4 --optimizer adam --batch_size 64 --noise_type asymmetric --noise_ratio 0.1
# negative_learning-pet-10%
python ./baseline_code/colearn/main_cvpr.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --uni_name NegativeLearning --learning_rate 1e-4 --optimizer adam --batch_size 64 --noise_type asymmetric --noise_ratio 0.1


# decoup-flower-10
python ./baseline_code/colearn/main_cvpr.py --model wideresnet50  --dataset flower-102 --num_epochs 10 --batch_size 64 --uni_name Decoupling --learning_rate 1e-4 --noise_type symmetric --noise_ratio 0.1
# decoup-pet-10
# python ./baseline_code/colearn/main_disc_cvpr.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --uni_name DISC --learning_rate 1e-3 --optimizer adam --batch_size 64 --noise_type asymmetric --noise_ratio 0.1

# disc-flower-10
python ./baseline_code/colearn/main_disc_cvpr.py --model wideresnet50  --dataset flower-102 --num_epochs 10 --batch_size 64 --uni_name DISC --learning_rate 1e-3 --noise_type symmetric --noise_ratio 0.1
# disc-pet-10
python ./baseline_code/colearn/main_disc_cvpr.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --uni_name DISC --learning_rate 1e-3 --optimizer adam --batch_size 64 --noise_type asymmetric --noise_ratio 0.1

# gjs-flower-10
python ./baseline_code/colearn/main_disc_cvpr.py --model wideresnet50  --dataset flower-102 --num_epochs 10 --batch_size 64 --uni_name GJS --learning_rate 1e-3 --noise_type symmetric --noise_ratio 0.1
# gjs-pet-10
python ./baseline_code/colearn/main_disc_cvpr.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --uni_name GJS --learning_rate 1e-3 --optimizer adam --batch_size 64 --noise_type asymmetric --noise_ratio 0.1

# elr-flower-10
python ./baseline_code/colearn/main_disc_cvpr.py --model wideresnet50  --dataset flower-102 --num_epochs 10 --batch_size 64 --uni_name ELR --learning_rate 1e-3 --noise_type symmetric --noise_ratio 0.1
# elr-pet-10
python ./baseline_code/colearn/main_disc_cvpr.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --uni_name ELR --learning_rate 1e-3 --optimizer adam --batch_size 64 --noise_type asymmetric --noise_ratio 0.1

