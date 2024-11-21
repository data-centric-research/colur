nohup python -u main_forget.py --save_dir ./outputs/resnet18_cifar10/retrain --unlearn retrain --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path ../data/cifar10/resnet18/retrain --shuffle > resnet18-cifar10-retrain.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_cifar10/FT --unlearn FT --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path ../data/cifar10/resnet18/FT --shuffle > resnet18-cifar10-FT.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_cifar10/GA --unlearn GA --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path ../data/cifar10/resnet18/GA --shuffle > resnet18-cifar10-GA.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_cifar10/FF --unlearn fisher --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path ../data/cifar10/resnet18/FF --shuffle > resnet18-cifar10-FF.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_cifar10/IU --unlearn wfisher --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path ../data/cifar10/resnet18/IU --shuffle > resnet18-cifar10-IU.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_cifar10/FT_prune --unlearn FT_prune --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --save_data --save_data_path ../data/cifar10/resnet18/FT_prune --shuffle > resnet18-cifar10-FT_prune.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_cifar10/retrain --unlearn retrain --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth  --resume --save_data --save_data_path ../data/cifar10/vgg16/retrain --shuffle > vgg16-cifar10-retrain.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_cifar10/FT --unlearn FT --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth  --resume --save_data --save_data_path ../data/cifar10/vgg16/FT --shuffle > vgg16-cifar10-FT.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_cifar10/GA --unlearn GA --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth  --resume --save_data --save_data_path ../data/cifar10/vgg16/GA --shuffle > vgg16-cifar10-GA.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_cifar10/IU --unlearn wfisher --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth  --resume --save_data --save_data_path ../data/cifar10/vgg16/IU --shuffle > vgg16-cifar10-IU.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_cifar10/FT_prune --unlearn FT_prune --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth  --resume --save_data --save_data_path ../data/cifar10/vgg16/FT_prune --shuffle > vgg16-cifar10-FT_prune.log 2>&1 &