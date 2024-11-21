 python -u main_forget.py --save_dir ./outputs/resnet18_cifar10/retrain --unlearn retrain --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --shuffle --eval_result_ft

 python -u main_forget.py --save_dir ./outputs/resnet18_cifar10/FT --unlearn FT --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --shuffle --eval_result_ft

 python -u main_forget.py --save_dir ./outputs/resnet18_cifar10/GA --unlearn GA --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --shuffle --eval_result_ft

 python -u main_forget.py --save_dir ./outputs/resnet18_cifar10/FF --unlearn fisher --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --shuffle --eval_result_ft

 python -u main_forget.py --save_dir ./outputs/resnet18_cifar10/IU --unlearn wfisher --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --shuffle --eval_result_ft

 python -u main_forget.py --save_dir ./outputs/resnet18_cifar10/FT_prune --unlearn FT_prune --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch resnet18 --resume --shuffle --eval_result_ft

 python -u main_forget.py --save_dir ./outputs/vgg16_cifar10/retrain --unlearn retrain --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth  --resume --shuffle --eval_result_ft

 python -u main_forget.py --save_dir ./outputs/vgg16_cifar10/FT --unlearn FT --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth  --resume --shuffle --eval_result_ft

 python -u main_forget.py --save_dir ./outputs/vgg16_cifar10/GA --unlearn GA --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth  --resume --shuffle --eval_result_ft

 python -u main_forget.py --save_dir ./outputs/vgg16_cifar10/IU --unlearn wfisher --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth  --resume --shuffle --eval_result_ft

 python -u main_forget.py --save_dir ./outputs/vgg16_cifar10/FT_prune --unlearn FT_prune --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --arch vgg16_bn_lth  --resume --shuffle --eval_result_ft