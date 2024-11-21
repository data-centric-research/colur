#!/bin/bash

# Define the GPUs to be used
GPUS=(0 1 2 3 4 5 6 7)

# Function to execute a training task on a specific GPU
run_training() {
    local gpu=$1
    local save_dir=$2
    local unlearn_method=$3
    local class_to_replace=$4
    local num_indexes_to_replace=$5
    local unlearn_epochs=$6
    local unlearn_lr=$7
    local arch=$8
    local additional_params=${9}
    local log_file=${10}

    CUDA_VISIBLE_DEVICES=$gpu nohup python -u main_forget.py \
        --data $DATASETS_BASE_DIR \
        --save_dir $save_dir \
        --unlearn $unlearn_method \
        --class_to_replace $class_to_replace \
        --num_indexes_to_replace $num_indexes_to_replace \
        --unlearn_epochs $unlearn_epochs \
        --unlearn_lr $unlearn_lr \
        --arch $arch \
        $additional_params \
        >$log_file 2>&1 &
}

# Training Dataset directories
DATASETS_BASE_DIR="/nvme/szh/data/3ai/lips/datasets"
TINYIMAGENET_DIR="$DATASETS_BASE_DIR/tiny-imagenet-200"

# Directory for saving the training outputs like Models and Logs
OUTPUT_BASE_DIR="/nvme/szh/data/3ai/lips/outputs"
OUTPUT_LOG_DIR_TRAIN_BACKBONE="/nvme/szh/data/3ai/lips/outputs/train-backbone-logs"

# Check if the TinyImagenet data directory exists
if [ ! -d "$TINYIMAGENET_DIR" ]; then
    echo "TinyImagenet data directory not found: $TINYIMAGENET_DIR"
    exit 1
fi

# Index to keep track of GPU allocation
GPU_INDEX=0
NUM_GPUS=${#GPUS[@]}

# 1.1.1 Resnet18 Cifar10
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/resnet18_cifar10/finetune_backbone" retrain 0 1 100 0.1 resnet18 "" "$OUTPUT_LOG_DIR_TRAIN_BACKBONE/Resnet18-Cifar10-bbft.log"
((GPU_INDEX++))

# 1.1.2 Resnet18 Cifar100
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/resnet18_cifar100/finetune_backbone" retrain 0 1 100 0.1 resnet18 "--dataset cifar100" "$OUTPUT_LOG_DIR_TRAIN_BACKBONE/Resnet18-Cifar100-bbft.log"
((GPU_INDEX++))

# 1.1.3 Resnet18 TinyImagenet
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/resnet18_tinyimg/finetune_backbone" retrain 0 1 100 0.1 resnet18 "--dataset TinyImagenet --data_dir $TINYIMAGENET_DIR" "$OUTPUT_LOG_DIR_TRAIN_BACKBONE/Resnet18-tinyim-bbft.log"
((GPU_INDEX++))

# 1.1.4 Resnet18 fashionMNIST
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/resnet18_fmnist/finetune_backbone" retrain 0 1 100 0.1 resnet18 "--dataset fashionMNIST" "$OUTPUT_LOG_DIR_TRAIN_BACKBONE/Resnet18-fsmnist-bbft.log"
((GPU_INDEX++))

# 1.2.1 vgg16_bn_lth Cifar10
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/vgg16_cifar10/finetune_backbone" retrain 0 1 100 0.1 vgg16_bn_lth "" "$OUTPUT_LOG_DIR_TRAIN_BACKBONE/VGG16-Cifar10-bbft.log"
((GPU_INDEX++))

# 1.2.2 vgg16_bn_lth Cifar100
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/vgg16_cifar100/finetune_backbone" retrain 0 1 100 0.1 vgg16_bn_lth "--dataset cifar100" "$OUTPUT_LOG_DIR_TRAIN_BACKBONE/VGG16-Cifar100-bbft.log"
((GPU_INDEX++))

# 1.2.3 vgg16_bn_lth TinyImagenet
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/vgg16_tinyimg/finetune_backbone" retrain 0 1 100 0.1 vgg16_bn_lth "--dataset TinyImagenet --data_dir $TINYIMAGENET_DIR" "$OUTPUT_LOG_DIR_TRAIN_BACKBONE/VGG16-tinyim-bbft.log"
((GPU_INDEX++))

# 1.2.4 vgg16_bn_lth fashionMNIST
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/vgg16_fmnist/finetune_backbone" retrain 0 1 100 0.1 vgg16_bn_lth "--dataset fashionMNIST" "$OUTPUT_LOG_DIR_TRAIN_BACKBONE/VGG16-fashionmn-bbft.log"
((GPU_INDEX++))

echo "All tasks have been started."
