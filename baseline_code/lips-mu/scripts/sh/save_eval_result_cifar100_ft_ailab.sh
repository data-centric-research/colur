# Directory for saving the training outputs like Models and Logs
OUTPUT_BASE_DIR="/nvme/szh/data/3ai/lips/outputs"

# Directory for saving machine unlearning results
SAVE_DATA_BASE_DIR="/nvme/szh/data/3ai/lips/saved_data"

if [ ! -d "$OUTPUT_BASE_DIR" ]; then
    echo "Directory for saving outputs not found: $OUTPUT_BASE_DIR"
    exit 1
fi
if [ ! -d "$SAVE_DATA_BASE_DIR" ]; then
    echo "Directory for saving MU data directory not found: $SAVE_DATA_BASE_DIR"
    exit 1
fi

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_cifar100/retrain --unlearn retrain --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar100/resnet18/retrain --shuffle --eval_result_ft

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_cifar100/FT --unlearn FT --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar100/resnet18/FT --shuffle --eval_result_ft

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_cifar100/GA --unlearn GA --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar100/resnet18/GA --shuffle --eval_result_ft

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_cifar100/FF --unlearn fisher --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar100/resnet18/FF --shuffle --eval_result_ft

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_cifar100/IU --unlearn wfisher --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar100/resnet18/IU --shuffle --eval_result_ft

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_cifar100/FT_prune --unlearn FT_prune --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar100/resnet18/FT_prune --shuffle --eval_result_ft

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_cifar100/retrain --unlearn retrain --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar100/vgg16/retrain --shuffle --eval_result_ft --dataset cifar100

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_cifar100/FT --unlearn FT --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar100/vgg16/FT --shuffle --eval_result_ft --dataset cifar100

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_cifar100/GA --unlearn GA --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar100/vgg16/GA --shuffle --eval_result_ft --dataset cifar100

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_cifar100/IU --unlearn wfisher --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar100/vgg16/IU --shuffle --eval_result_ft --dataset cifar100

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_cifar100/FT_prune --unlearn FT_prune --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar100/vgg16/FT_prune --shuffle --eval_result_ft --dataset cifar100