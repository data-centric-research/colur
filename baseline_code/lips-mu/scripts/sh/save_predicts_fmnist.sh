nohup python -u main_forget.py --save_dir ./outputs/resnet18_fmnist/retrain --unlearn retrain --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --save_data --save_data_path ../data/fmnist/resnet18/retrain --shuffle > resnet18-fmnist-retrain.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_fmnist/FT --unlearn FT --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --save_data --save_data_path ../data/fmnist/resnet18/FT --shuffle > resnet18-fmnist-FT.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_fmnist/GA --unlearn GA --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --save_data --save_data_path ../data/fmnist/resnet18/GA --shuffle > resnet18-fmnist-GA.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_fmnist/FF --unlearn fisher --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --save_data --save_data_path ../data/fmnist/resnet18/FF --shuffle > resnet18-fmnist-FF.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_fmnist/IU --unlearn wfisher --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --save_data --save_data_path ../data/fmnist/resnet18/IU --shuffle > resnet18-fmnist-IU.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/resnet18_fmnist/FT_prune --unlearn FT_prune --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --save_data --save_data_path ../data/fmnist/resnet18/FT_prune --shuffle > resnet18-fmnist-FT_prune.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_fmnist/retrain --unlearn retrain --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --save_data --save_data_path ../data/fmnist/vgg16/retrain --shuffle > vgg16-fmnist-retrain.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_fmnist/FT --unlearn FT --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --save_data --save_data_path ../data/fmnist/vgg16/FT --shuffle > vgg16-fmnist-FT.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_fmnist/GA --unlearn GA --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --save_data --save_data_path ../data/fmnist/vgg16/GA --shuffle > vgg16-fmnist-GA.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_fmnist/IU --unlearn wfisher --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --save_data --save_data_path ../data/fmnist/vgg16/IU --shuffle > vgg16-fmnist-IU.log 2>&1 &

nohup python -u main_forget.py --save_dir ./outputs/vgg16_fmnist/FT_prune --unlearn FT_prune --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --save_data --save_data_path ../data/fmnist/vgg16/FT_prune --shuffle > vgg16-fmnist-FT_prune.log 2>&1 &