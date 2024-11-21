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

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_cifar10/retrain --unlearn retrain --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar10/resnet18/retrain --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_cifar10/FT --unlearn FT --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar10/resnet18/FT --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_cifar10/GA --unlearn GA --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar10/resnet18/GA --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_cifar10/FF --unlearn fisher --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar10/resnet18/FF --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_cifar10/IU --unlearn wfisher --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar10/resnet18/IU --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_cifar10/FT_prune --unlearn FT_prune --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar10/resnet18/FT_prune --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_cifar10/retrain --unlearn retrain --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar10/vgg16/retrain --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_cifar10/FT --unlearn FT --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar10/vgg16/FT --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_cifar10/GA --unlearn GA --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar10/vgg16/GA --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_cifar10/IU --unlearn wfisher --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar10/vgg16/IU --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_cifar10/FT_prune --unlearn FT_prune --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/cifar10/vgg16/FT_prune --shuffle
