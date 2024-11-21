#!/bin/bash

# 检查是否传递了至少一个GPU的参数
if [ "$#" -lt 1 ]; then
    echo "使用方法: ./this_script.sh <GPU_ID_1> <GPU_ID_2> ... <GPU_ID_N>"
    exit 1
fi

# 获取传递的所有 GPU ID，并将其存储在一个数组中
GPU_IDS=("$@")
echo "Available GPUs: ${GPU_IDS[@]}"

# 设置PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# 定义要运行的脚本
declare -a scripts=(
    # "run_cifar10_mu_ft_cvpr.sh" # 3
    # "run_cifar10_mu_ga_cvpr.sh" # 4
    "run_cifar10_mu_gal1_cvpr.sh" # 3
    "run_cifar10_mu_wfisher_cvpr.sh" # 4
    # # "run_cifar10_mu_ftl1_cvpr.sh" # 5
    # # "run_cifar10_mu_ftprune_cvpr.sh" # 8
    # # "run_cifar10_mu_ftprunebi_cvpr.sh" # 9
    # # "run_cifar10_mu_retrain_cvpr.sh" # 10
    # # "run_cifar10_mu_retrainls_cvpr.sh" # 11
    # # "run_cifar10_mu_retrainsam_cvpr.sh" # 12
    # # "run_cifar10_mu_raw_cvpr.sh" # 1
    # # "run_cifar10_mu_fisher_new_cvpr.sh" # 6
)

# 检查GPU数量是否足够
if [ "${#GPU_IDS[@]}" -lt "${#scripts[@]}" ]; then
    echo "Error: Not enough GPUs provided. Required: ${#scripts[@]}, Provided: ${#GPU_IDS[@]}"
    exit 1
fi

# 循环分配 GPU 给每个脚本
for i in "${!scripts[@]}"; do
    GPU_ID=${GPU_IDS[$i]}
    echo "Assigning ${scripts[$i]} to GPU $GPU_ID"

    CUDA_VISIBLE_DEVICES=$GPU_ID nohup bash "${scripts[$i]}" $GPU_ID >"logs/${scripts[$i]}.log" 2>&1 &
done

echo "All scripts are running in parallel on specified GPUs."
