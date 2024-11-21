# 设置指定的 GPU
export CUDA_VISIBLE_DEVICES=$1
echo "Using GPU $1"
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# Pretrain:
python ./run_experiment_cvpr.py --model wideresnet50 --dataset flower-102 --num_epochs 20 --train_mode pretrain --learning_rate 1e-3 --optimizer adam --batch_size 256 --noise_type symmetric --noise_ratio 0.5

# Incremental Train:
python ./run_experiment_cvpr.py --model wideresnet50 --dataset flower-102 --num_epochs 15 --train_mode inc_train --learning_rate 1e-3 --optimizer adam --batch_size 256 --noise_type symmetric --noise_ratio 0.5
