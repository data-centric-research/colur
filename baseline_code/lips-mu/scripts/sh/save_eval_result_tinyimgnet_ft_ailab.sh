# Directory for saving the training outputs like Models and Logs
OUTPUT_BASE_DIR="/nvme/szh/data/3ai/lips/outputs"

# Directory for saving machine unlearning results
SAVE_DATA_BASE_DIR="/nvme/szh/data/3ai/lips/saved_data"

DATASET_DIR="/nvme/szh/data/3ai/lips/datasets"

if [ ! -d "$OUTPUT_BASE_DIR" ]; then
    echo "Directory for saving outputs not found: $OUTPUT_BASE_DIR"
    exit 1
fi
if [ ! -d "$SAVE_DATA_BASE_DIR" ]; then
    echo "Directory for saving MU data directory not found: $SAVE_DATA_BASE_DIR"
    exit 1
fi

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_tinyimg/retrain --unlearn retrain --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir $DATASET_DIR/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/resnet18/retrain --shuffle --eval_result_ft

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_tinyimg/FT --unlearn FT --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir $DATASET_DIR/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/resnet18/FT --shuffle --eval_result_ft

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_tinyimg/GA --unlearn GA --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir $DATASET_DIR/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/resnet18/GA --shuffle --eval_result_ft

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_tinyimg/FF --unlearn fisher --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir $DATASET_DIR/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/resnet18/FF --shuffle --eval_result_ft

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_tinyimg/IU --unlearn wfisher --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir $DATASET_DIR/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/resnet18/IU --shuffle --eval_result_ft

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_tinyimg/FT_prune --unlearn FT_prune --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250  --dataset TinyImagenet --data_dir $DATASET_DIR/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/resnet18/FT_prune --shuffle --eval_result_ft

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_tinyimg/retrain --unlearn retrain --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir $DATASET_DIR/tiny-imagenet-200 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/vgg16/retrain --shuffle --eval_result_ft

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_tinyimg/FT --unlearn FT --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir $DATASET_DIR/tiny-imagenet-200 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/vgg16/FT --shuffle --eval_result_ft

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_tinyimg/GA --unlearn GA --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir $DATASET_DIR/tiny-imagenet-200 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/vgg16/GA --shuffle --eval_result_ft

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_tinyimg/IU --unlearn wfisher --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir $DATASET_DIR/tiny-imagenet-200 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/vgg16/IU --shuffle --eval_result_ft

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_tinyimg/FT_prune --unlearn FT_prune --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir $DATASET_DIR/tiny-imagenet-200 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/tinyimgnet/vgg16/FT_prune --shuffle --eval_result_ft