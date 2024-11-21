DATASETS_BASE_DIR="/nvme/szh/data/3ai/lips/datasets"
TINYIMAGENET_DIR="$DATASETS_BASE_DIR/tiny-imagenet-200"

# Check if the TinyImagenet data directory exists
if [ ! -d "$TINYIMAGENET_DIR" ]; then
    echo "TinyImagenet data directory not found: $TINYIMAGENET_DIR"
    exit 1
fi

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

nohup python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_tinyimg/retrain --unlearn retrain --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir $TINYIMAGENET_DIR --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/resnet18/retrain --shuffle > resnet18-tinyimg-retrain.log 2>&1 &

nohup python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_tinyimg/FT --unlearn FT --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir $TINYIMAGENET_DIR --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/resnet18/FT --shuffle > resnet18-tinyimg-FT.log 2>&1 &

nohup python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_tinyimg/GA --unlearn GA --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir $TINYIMAGENET_DIR --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/resnet18/GA --shuffle > resnet18-tinyimg-GA.log 2>&1 &

nohup python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_tinyimg/FF --unlearn fisher --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir $TINYIMAGENET_DIR --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/resnet18/FF --shuffle > resnet18-tinyimg-FF.log 2>&1 &

nohup python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_tinyimg/IU --unlearn wfisher --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir $TINYIMAGENET_DIR --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/resnet18/IU --shuffle > resnet18-tinyimg-IU.log 2>&1 &

nohup python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_tinyimg/FT_prune --unlearn FT_prune --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250  --dataset TinyImagenet --data_dir $TINYIMAGENET_DIR --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/resnet18/FT_prune --shuffle > resnet18-tinyimg-FT_prune.log 2>&1 &

nohup python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_tinyimg/retrain --unlearn retrain --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir $TINYIMAGENET_DIR --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/vgg16/retrain --shuffle > vgg16-tinyimg-retrain.log 2>&1 &

nohup python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_tinyimg/FT --unlearn FT --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir $TINYIMAGENET_DIR --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/vgg16/FT --shuffle > vgg16-tinyimg-FT.log 2>&1 &

nohup python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_tinyimg/GA --unlearn GA --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir $TINYIMAGENET_DIR --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/vgg16/GA --shuffle > vgg16-tinyimg-GA.log 2>&1 &

nohup python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_tinyimg/IU --unlearn wfisher --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir $TINYIMAGENET_DIR --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/vgg16/IU --shuffle > vgg16-tinyimg-IU.log 2>&1 &

nohup python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_tinyimg/FT_prune --unlearn FT_prune --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir $TINYIMAGENET_DIR --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/vgg16/FT_prune --shuffle > vgg16-tinyimg-FT_prune.log 2>&1 &