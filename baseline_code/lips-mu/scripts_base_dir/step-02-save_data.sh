BASE_LOG_DIR="/nvme/szh/data/3ai/lips/logs"
STEP_02_SAVE_DATA_LOG_DIR="$BASE_LOG_DIR/step-02-save_data"

if [ ! -d "$STEP_02_SAVE_DATA_LOG_DIR" ]; then
  mkdir -p $STEP_02_SAVE_DATA_LOG_DIR
fi

CUDA_VISIBLE_DEVICES=1 nohup sh scripts/sh/save_predicts_cifar10_ailab.sh > $STEP_02_SAVE_DATA_LOG_DIR/save_predicts_cifar10_ailab.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh scripts/sh/save_predicts_cifar100_ailab.sh > $STEP_02_SAVE_DATA_LOG_DIR/save_predicts_cifar100_ailab.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh scripts/sh/save_predicts_fmnist_ailab.sh > $STEP_02_SAVE_DATA_LOG_DIR/save_predicts_fmnist_ailab.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh scripts/sh/save_predicts_tinyimgnet_ailab.sh > $STEP_02_SAVE_DATA_LOG_DIR/save_predicts_tinyimgnet_ailab.log 2>&1 &