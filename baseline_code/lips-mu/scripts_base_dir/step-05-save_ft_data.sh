BASE_LOG_DIR="/nvme/szh/data/3ai/lips/logs"
STEP_05_SAVE_DATA_LOG_DIR="$BASE_LOG_DIR/step-05-save_ft_data"

if [ ! -d "$STEP_05_SAVE_DATA_LOG_DIR" ]; then
    mkdir -p $STEP_05_SAVE_DATA_LOG_DIR
fi
# Define the GPUs to be used
CUDA_VISIBLE_DEVICES=2 nohup sh scripts/sh/save_eval_result_cifar10_ft_ailab.sh >$STEP_05_SAVE_DATA_LOG_DIR/save_eval_result_cifar10_ft_ailab.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh scripts/sh/save_eval_result_cifar100_ft_ailab.sh >$STEP_05_SAVE_DATA_LOG_DIR/save_eval_result_cifar100_ft_ailab.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh scripts/sh/save_eval_result_fmnist_ft_ailab.sh >$STEP_05_SAVE_DATA_LOG_DIR/save_eval_result_fmnist_ft_ailab.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh scripts/sh/save_eval_result_tinyimgnet_ft_ailab.sh >$STEP_05_SAVE_DATA_LOG_DIR/save_eval_result_tinyimgnet_ft_ailab.log 2>&1 &
