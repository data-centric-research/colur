nohup python -u main_forget.py --save_dir ./outputs/resnet18_tinyimg/retrain --unlearn retrain --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path ../data/tinyimgnet/resnet18/retrain --shuffle > resnet18-tinyimg-retrain.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_tinyimg/FT --unlearn FT --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path ../data/tinyimgnet/resnet18/FT --shuffle > resnet18-tinyimg-FT.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_tinyimg/GA --unlearn GA --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path ../data/tinyimgnet/resnet18/GA --shuffle > resnet18-tinyimg-GA.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_tinyimg/FF --unlearn fisher --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path ../data/tinyimgnet/resnet18/FF --shuffle > resnet18-tinyimg-FF.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_tinyimg/IU --unlearn wfisher --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path ../data/tinyimgnet/resnet18/IU --shuffle > resnet18-tinyimg-IU.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_tinyimg/FT_prune --unlearn FT_prune --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250  --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch resnet18 --resume --save_data --save_data_path ../data/tinyimgnet/resnet18/FT_prune --shuffle > resnet18-tinyimg-FT_prune.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_tinyimg/retrain --unlearn retrain --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --resume --save_data --save_data_path ../data/tinyimgnet/vgg16/retrain --shuffle > vgg16-tinyimg-retrain.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_tinyimg/FT --unlearn FT --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --resume --save_data --save_data_path ../data/tinyimgnet/vgg16/FT --shuffle > vgg16-tinyimg-FT.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_tinyimg/GA --unlearn GA --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --resume --save_data --save_data_path ../data/tinyimgnet/vgg16/GA --shuffle > vgg16-tinyimg-GA.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_tinyimg/IU --unlearn wfisher --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --resume --save_data --save_data_path ../data/tinyimgnet/vgg16/IU --shuffle > vgg16-tinyimg-IU.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_tinyimg/FT_prune --unlearn FT_prune --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --arch vgg16_bn_lth  --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --resume --save_data --save_data_path ../data/tinyimgnet/vgg16/FT_prune --shuffle > vgg16-tinyimg-FT_prune.log 2>&1 &