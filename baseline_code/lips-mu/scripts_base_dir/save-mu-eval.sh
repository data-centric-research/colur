# Directory for saving machine unlearning results
BASE_SAVE_DATA_DIR="/nvme/szh/data/3ai/lips/saved_data"
if [ ! -d "$BASE_SAVE_DATA_DIR" ]; then
    echo "Directory for saving MU data directory not found: $BASE_SAVE_DATA_DIR"
    exit 1
fi
# Directory for saving the training outputs like Models and Logs
BASE_OUTPUT_DIR="/nvme/szh/data/3ai/lips/outputs"

if [ ! -d "$BASE_OUTPUT_DIR" ]; then
    echo "Directory for saving outputs not found: $BASE_OUTPUT_DIR"
    exit 1
fi

#2.1 Cifar-10

# RESNET18
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_cifar10/retrain --unlearn retrain --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar10/resnet18/retrain --shuffle
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_cifar10/FT --unlearn FT --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar10/resnet18/FT --shuffle
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_cifar10/GA --unlearn GA --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar10/resnet18/GA --shuffle
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_cifar10/FF --unlearn fisher --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar10/resnet18/FF --shuffle
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_cifar10/IU --unlearn wfisher --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar10/resnet18/IU --shuffle
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_cifar10/FT_prune --unlearn FT_prune --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar10/resnet18/FT_prune --shuffle

# VGG16
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_cifar10/retrain --unlearn retrain --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth  --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar10/vgg16/retrain --shuffle
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_cifar10/FT --unlearn FT --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth  --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar10/vgg16/FT --shuffle
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_cifar10/GA --unlearn GA --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth  --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar10/vgg16/GA --shuffle
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_cifar10/IU --unlearn wfisher --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth  --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar10/vgg16/IU --shuffle
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_cifar10/FT_prune --unlearn FT_prune --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth  --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar10/vgg16/FT_prune --shuffle

#2.2 Cifar-100
# RESNET18
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_cifar100/retrain --unlearn retrain --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar100/resnet18/retrain --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_cifar100/FT --unlearn FT --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar100/resnet18/FT --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_cifar100/GA --unlearn GA --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar100/resnet18/GA --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_cifar100/FF --unlearn fisher --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar100/resnet18/FF --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_cifar100/IU --unlearn wfisher --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar100/resnet18/IU --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_cifar100/FT_prune --unlearn FT_prune --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --dataset cifar100 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar100/resnet18/FT_prune --shuffle

# VGG16
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_cifar100/retrain --unlearn retrain --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar100/vgg16/retrain --shuffle --dataset cifar100

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_cifar100/FT --unlearn FT --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar100/vgg16/FT --shuffle --dataset cifar100

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_cifar100/GA --unlearn GA --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar100/vgg16/GA --shuffle --dataset cifar100

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_cifar100/IU --unlearn wfisher --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar100/vgg16/IU --shuffle --dataset cifar100

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_cifar100/FT_prune --unlearn FT_prune --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --arch vgg16_bn_lth  --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/cifar100/vgg16/FT_prune --shuffle --dataset cifar100

#2.3 TinyImageNet

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_tinyimg/retrain --unlearn retrain --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir $BASE_SAVE_DATA_DIR/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/tinyimgnet/resnet18/retrain --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_tinyimg/FT --unlearn FT --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir $BASE_SAVE_DATA_DIR/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/tinyimgnet/resnet18/FT --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_tinyimg/GA --unlearn GA --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir $BASE_SAVE_DATA_DIR/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/tinyimgnet/resnet18/GA --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_tinyimg/FF --unlearn fisher --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir $BASE_SAVE_DATA_DIR/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/tinyimgnet/resnet18/FF --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_tinyimg/IU --unlearn wfisher --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir $BASE_SAVE_DATA_DIR/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/tinyimgnet/resnet18/IU --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_tinyimg/FT_prune --unlearn FT_prune --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250  --dataset TinyImagenet --data_dir $BASE_SAVE_DATA_DIR/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/tinyimgnet/resnet18/FT_prune --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_tinyimg/retrain --unlearn retrain --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir $BASE_SAVE_DATA_DIR/tiny-imagenet-200 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/tinyimgnet/vgg16/retrain --shuffle
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_tinyimg/FT --unlearn FT --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir $BASE_SAVE_DATA_DIR/tiny-imagenet-200 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/tinyimgnet/vgg16/FT --shuffle
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_tinyimg/GA --unlearn GA --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir $BASE_SAVE_DATA_DIR/tiny-imagenet-200 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/tinyimgnet/vgg16/GA --shuffle
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_tinyimg/IU --unlearn wfisher --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir $BASE_SAVE_DATA_DIR/tiny-imagenet-200 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/tinyimgnet/vgg16/IU --shuffle
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_tinyimg/FT_prune --unlearn FT_prune --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir $BASE_SAVE_DATA_DIR/tiny-imagenet-200 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/tinyimgnet/vgg16/FT_prune --shuffle

#2.4 FashionMNIST

# RESNET18
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_fmnist/retrain --unlearn retrain --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/fmnist/resnet18/retrain --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_fmnist/FT --unlearn FT --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/fmnist/resnet18/FT --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_fmnist/GA --unlearn GA --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/fmnist/resnet18/GA --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_fmnist/FF --unlearn fisher --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/fmnist/resnet18/FF --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_fmnist/IU --unlearn wfisher --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/fmnist/resnet18/IU --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/resnet18_fmnist/FT_prune --unlearn FT_prune --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/fmnist/resnet18/FT_prune --shuffle

# VGG16
python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_fmnist/retrain --unlearn retrain --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/fmnist/vgg16/retrain --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_fmnist/FT --unlearn FT --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/fmnist/vgg16/FT --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_fmnist/GA --unlearn GA --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/fmnist/vgg16/GA --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_fmnist/IU --unlearn wfisher --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/fmnist/vgg16/IU --shuffle

python -u main_forget.py --save_dir $BASE_OUTPUT_DIR/vgg16_fmnist/FT_prune --unlearn FT_prune --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --save_data --save_data_path $BASE_SAVE_DATA_DIR/fmnist/vgg16/FT_prune --shuffle