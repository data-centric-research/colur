nohup python -u main_forget.py --save_dir ./outputs/resnet18_fmnist/retrain --unlearn retrain --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --shuffle --eval_result_ft

nohup python -u main_forget.py --save_dir ./outputs/resnet18_fmnist/FT --unlearn FT --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --shuffle --eval_result_ft

nohup python -u main_forget.py --save_dir ./outputs/resnet18_fmnist/GA --unlearn GA --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --shuffle --eval_result_ft

python -u main_forget.py --save_dir ./outputs/resnet18_fmnist/FF --unlearn fisher --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --shuffle --eval_result_ft

python -u main_forget.py --save_dir ./outputs/resnet18_fmnist/IU --unlearn wfisher --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --shuffle --eval_result_ft

python -u main_forget.py --save_dir ./outputs/resnet18_fmnist/FT_prune --unlearn FT_prune --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST  --arch resnet18 --resume --shuffle --eval_result_ft

python -u main_forget.py --save_dir ./outputs/vgg16_fmnist/retrain --unlearn retrain --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --shuffle --eval_result_ft

python -u main_forget.py --save_dir ./outputs/vgg16_fmnist/FT --unlearn FT --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --shuffle --eval_result_ft

python -u main_forget.py --save_dir ./outputs/vgg16_fmnist/GA --unlearn GA --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --shuffle --eval_result_ft

python -u main_forget.py --save_dir ./outputs/vgg16_fmnist/IU --unlearn wfisher --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --shuffle --eval_result_ft

python -u main_forget.py --save_dir ./outputs/vgg16_fmnist/FT_prune --unlearn FT_prune --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --shuffle --eval_result_ft