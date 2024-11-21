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
        > $log_file 2>&1 &
}

# Training Dataset directories
DATASETS_BASE_DIR="/nvme/szh/data/3ai/lips/datasets"
TINYIMAGENET_DIR="$DATASETS_BASE_DIR/tiny-imagenet-200"

# Directory for saving the training outputs like Models and Logs
OUTPUT_BASE_DIR="/nvme/szh/data/3ai/lips/outputs"

OUTPUT_LOG_DIR_TRAIN_MU="/nvme/szh/data/3ai/lips/outputs/mu-training-logs"

# Check if the TinyImagenet data directory exists
if [ ! -d "$TINYIMAGENET_DIR" ]; then
    echo "TinyImagenet data directory not found: $TINYIMAGENET_DIR"
    exit 1
fi

# Index to keep track of GPU allocation
GPU_INDEX=0
NUM_GPUS=${#GPUS[@]}



# Original
# run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/resnet18_tinyimg/FF" fisher 1,51,101,151,198 250 100 0.1 resnet18 "--dataset TinyImagenet --data_dir $TINYIMAGENET_DIR --load_ff --resume --alpha 10.2" "$OUTPUT_LOG_DIR_TRAIN_MU/Resnet18-tinyim-ff.log"
# ((GPU_INDEX++))
# Modified  1
run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/resnet18_tinyimg/FF" fisher 1,51,101,151,198 250 100 0.01 resnet18 "--dataset TinyImagenet --data_dir $TINYIMAGENET_DIR --load_ff --resume --alpha 20.4" "$OUTPUT_LOG_DIR_TRAIN_MU/Resnet18-tinyim-ff.log"
((GPU_INDEX++))

# Original
# run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/resnet18_tinyimg/FT_prune" FT_prune 1,51,101,151,198 250 4 0.000005 resnet18 "--dataset TinyImagenet --data_dir $TINYIMAGENET_DIR --load_ff --resume" "$OUTPUT_LOG_DIR_TRAIN_MU/Resnet18-tinyim-ft-prune.log"
# ((GPU_INDEX++))
# Modified  2
# run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/resnet18_tinyimg/FT_prune" FT_prune 1,51,101,151,198 250 4 0.0000005 resnet18 "--dataset TinyImagenet --data_dir $TINYIMAGENET_DIR --load_ff --resume" "$OUTPUT_LOG_DIR_TRAIN_MU/Resnet18-tinyim-ft-prune.log"
# ((GPU_INDEX++))

# Original
# run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/vgg16_cifar10/IU" wfisher 0,1,3,5,6 2250 100 0.1 vgg16_bn_lth "--load_ff --resume --alpha 1" "$OUTPUT_LOG_DIR_TRAIN_MU/VGG16-Cifar10-iu.log"
# ((GPU_INDEX++))
# Modified  3
# run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/vgg16_cifar10/IU" wfisher 0,1,3,5,6 2250 100 0.01 vgg16_bn_lth "--load_ff --resume --alpha 2" "$OUTPUT_LOG_DIR_TRAIN_MU/VGG16-Cifar10-iu.log"
# ((GPU_INDEX++))


# Original
# run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/vgg16_cifar100/IU" wfisher 11,22,33,44,55 225 100 0.1 vgg16_bn_lth "--dataset cifar100 --load_ff --resume --alpha 6" "$OUTPUT_LOG_DIR_TRAIN_MU/VGG16-Cifar100-iu.log"
# ((GPU_INDEX++))
# Modified  4
# run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/vgg16_cifar100/IU" wfisher 11,22,33,44,55 225 100 0.01 vgg16_bn_lth "--dataset cifar100 --load_ff --resume --alpha 12" "$OUTPUT_LOG_DIR_TRAIN_MU/VGG16-Cifar100-iu.log"
# ((GPU_INDEX++))

# Original
# run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/vgg16_cifar100/FT_prune" FT_prune 11,22,33,44,55 225 20 0.000005 vgg16_bn_lth "--dataset cifar100 --load_ff --resume" "$OUTPUT_LOG_DIR_TRAIN_MU/VGG16-Cifar100-ft-prune.log"
# ((GPU_INDEX++))
# Modified  5
# run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/vgg16_cifar100/FT_prune" FT_prune 11,22,33,44,55 225 20 0.0000005 vgg16_bn_lth "--dataset cifar100 --load_ff --resume" "$OUTPUT_LOG_DIR_TRAIN_MU/VGG16-Cifar100-ft-prune.log"
# ((GPU_INDEX++))


# Original
# run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/vgg16_tinyimg/IU" wfisher 1,51,101,151,198 250 100 0.1 vgg16_bn_lth "--dataset TinyImagenet --data_dir $TINYIMAGENET_DIR --load_ff --resume --alpha 10" "$OUTPUT_LOG_DIR_TRAIN_MU/VGG16-tinyim-iu.log"
# ((GPU_INDEX++))
# Modified  6
# run_training ${GPUS[GPU_INDEX % NUM_GPUS]} "$OUTPUT_BASE_DIR/vgg16_tinyimg/IU" wfisher 1,51,101,151,198 250 100 0.01 vgg16_bn_lth "--dataset TinyImagenet --data_dir $TINYIMAGENET_DIR --load_ff --resume --alpha 20" "$OUTPUT_LOG_DIR_TRAIN_MU/VGG16-tinyim-iu.log"
# ((GPU_INDEX++))

echo "All tasks have been started."
