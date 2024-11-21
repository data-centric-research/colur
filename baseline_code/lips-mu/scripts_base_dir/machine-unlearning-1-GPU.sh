BASE_DIR="/nvme/szh/data/3ai/lips/"


# 1.1 Resnet18 Unlearning Training

# 1.1.1 Resnet18 Cifar10

# (0) backbone_ft
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_cifar10/finetune_backbone --unlearn retrain --class_to_replace 0 --num_indexes_to_replace 1 --unlearn_epochs 100 --unlearn_lr 0.1 --arch resnet18 > Resnet18-Cifar10-bbft.log 2>&1

# (1) retrain
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_cifar10/retrain --unlearn retrain --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --unlearn_epochs 100 --unlearn_lr 0.1 --arch resnet18 > Resnet18-Cifar10-retrain.log 2>&1

# (2) FT
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_cifar10/FT --unlearn FT --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --unlearn_epochs 100 --unlearn_lr 0.1 --arch resnet18 --load_ff --resume > Resnet18-Cifar10-ft.log 2>&1 &

# (3) GA
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_cifar10/GA --unlearn GA --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --unlearn_epochs 4 --unlearn_lr 0.0001 --arch resnet18 --load_ff --resume > Resnet18-Cifar10-ga.log 2>&1 &

# (4) FF
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_cifar10/FF --unlearn fisher --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --unlearn_epochs 100 --unlearn_lr 0.1 --arch resnet18 --load_ff –resume --alpha  16.5 > Resnet18-Cifar10-ff.log 2>&1 &

# (5) IU
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_cifar10/IU --unlearn wfisher --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --unlearn_epochs 100 --unlearn_lr 0.1 --arch resnet18 --load_ff --resume --alpha 16 > Resnet18-Cifar10-iu.log 2>&1 &

# (6) FT_prune
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_cifar10/FT_prune --unlearn FT_prune --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --unlearn_epochs 30 --unlearn_lr 0.00001 --arch resnet18 --load_ff --resume > Resnet18-Cifar10-ft-prune.log 2>&1 &

# 1.1.2 Resnet18 Cifar100
# (0) backbone_ft
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_cifar100/finetune_backbone --unlearn retrain --class_to_replace 0 --num_indexes_to_replace 1 --unlearn_epochs 100 --unlearn_lr 0.1 --arch resnet18 --dataset cifar100 > Resnet18-Cifar100-bbft.log 2>&1 &

# (1) retrain
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_cifar100/retrain --unlearn retrain --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --unlearn_epochs 100 --unlearn_lr 0.1 --arch resnet18 --dataset cifar100 > Resnet18-Cifar100-retrain.log 2>&1 &

# (2) FT
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_cifar100/FT --unlearn FT --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --unlearn_epochs 100 --unlearn_lr 0.1 --arch resnet18 --dataset cifar100 --load_ff --resume > Resnet18-Cifar100-ft.log 2>&1 &

# (3) GA
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_cifar100/GA --unlearn GA --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --unlearn_epochs 4 --unlearn_lr 0.001 --arch resnet18 --dataset cifar100 --load_ff --resume > Resnet18-Cifar100-ga.log 2>&1 &

# (4) FF
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_cifar100/FF --unlearn fisher --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --unlearn_epochs 100 --unlearn_lr 0.1 --arch resnet18 --dataset cifar100 --load_ff --resume --alpha 20 > Resnet18-Cifar100-ff.log 2>&1 &

# (5) IU
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_cifar100/IU --unlearn wfisher --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --unlearn_epochs 100 --unlearn_lr 0.1 --arch resnet18 --dataset cifar100 --load_ff --resume --alpha 160 > Resnet18-Cifar100-iu.log 2>&1 &

# (6) FT_prune
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_cifar100/FT_prune --unlearn FT_prune --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --unlearn_epochs 20 --unlearn_lr 0.00001 --arch resnet18 --dataset cifar100 --load_ff --resume > Resnet18-Cifar100-ft-prune.log 2>&1 &

# 1.1.3 Resnet18 TinyImagenet
# (0) backbone_ft
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_tinyimg/finetune_backbone --unlearn retrain --class_to_replace 0 --num_indexes_to_replace 1 --unlearn_epochs 100 --unlearn_lr 0.1 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch resnet18 --batch_size 128 > Resnet18-tinyim-bbft.log 2>&1 &

# (1) retrain
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_tinyimg/retrain --unlearn retrain --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --unlearn_epochs 100 --unlearn_lr 0.1 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch resnet18 --batch_size 128 > Resnet18-tinyim-retrain.log 2>&1 &

# (2) FT
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_tinyimg/FT --unlearn FT --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --unlearn_epochs 100 --unlearn_lr 0.1 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch resnet18 --batch_size 128  --load_ff --resume > Resnet18-tinyim-ft.log 2>&1 &

# (3) GA
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_tinyimg/GA --unlearn GA --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --unlearn_epochs 5 --unlearn_lr 0.00001 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch resnet18 --batch_size 128  --load_ff --resume > Resnet18-tinyim-ga.log 2>&1 &

# (4) FF
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_tinyimg/FF --unlearn fisher --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --unlearn_epochs 100 --unlearn_lr 0.1 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch resnet18 --batch_size 128  --load_ff --resume --alpha 10.2 > Resnet18-tinyim-ff.log 2>&1 &

# (5) IU
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_tinyimg/IU --unlearn wfisher --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --unlearn_epochs 100 --unlearn_lr 0.1 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch resnet18 --batch_size 128  --load_ff --resume --alpha 160 > Resnet18-tinyim-iu.log 2>&1 &

# (6) FT_prune
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_tinyimg/FT_prune --unlearn FT_prune --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --unlearn_epochs 4 --unlearn_lr 0.000005 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch resnet18 --batch_size 128  --load_ff --resume > Resnet18-tinyim-ft-prune.log 2>&1 &

# 1.1.4 Resnet18 fashionMNIST
# (0) backbone_ft
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_fmnist/finetune_backbone --unlearn retrain --class_to_replace 0 --num_indexes_to_replace 1 --unlearn_epochs 100 --unlearn_lr 0.1 --dataset fashionMNIST > Resnet18-fsmnist-bbft.log 2>&1 &

# (1) retrain
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_fmnist/retrain --unlearn retrain --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700  --unlearn_epochs 100 --unlearn_lr 0.1 --dataset fashionMNIST > Resnet18-fsmnist-retrain.log 2>&1 &

# (2) FT
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_fmnist/FT --unlearn FT --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700  --unlearn_epochs 100 --unlearn_lr 0.1 --dataset fashionMNIST --load_ff --resume > Resnet18-fsmnist-ft.log 2>&1 &

# (3) GA
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_fmnist/GA --unlearn GA --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700  --unlearn_epochs 5 --unlearn_lr 0.00001 --dataset fashionMNIST --load_ff --resume > Resnet18-fsmnist-ga.log 2>&1 &

# (4) FF
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_fmnist/FF --unlearn fisher --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700  --unlearn_epochs 100 --unlearn_lr 0.1 --dataset fashionMNIST --load_ff --resume --alpha 16.5 > Resnet18-fsmnist-ff.log 2>&1 &

# (5) IU
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_fmnist/IU --unlearn wfisher --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700  --unlearn_epochs 100 --unlearn_lr 0.1 --dataset fashionMNIST --load_ff --resume --alpha 60 > Resnet18-fsmnist-iu.log 2>&1 &

# (6) FT_prune
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/resnet18_fmnist/FT_prune --unlearn FT_prune --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700  --unlearn_epochs 20 --unlearn_lr 0.00001 --dataset fashionMNIST --load_ff --resume > Resnet18-fsmnist-ft-prune.log 2>&1 &

# 1.2 VGG16 Unlearning Training
# 1.2.1 vgg16_bn_lth Cifar10
# (0) backbone_ft
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_cifar10/finetune_backbone --unlearn retrain --class_to_replace 0 --num_indexes_to_replace 1 --unlearn_epochs 100 --unlearn_lr 0.1 --arch vgg16_bn_lth > VGG16-Cifar10-bbft.log 2>&1 &

# (1) retrain
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_cifar10/retrain --unlearn retrain --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --unlearn_epochs 100 --unlearn_lr 0.1 --arch vgg16_bn_lth > VGG16-Cifar10-retrain.log 2>&1 &

# (2) FT
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_cifar10/FT --unlearn FT --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --unlearn_epochs 100 --unlearn_lr 0.1 --arch vgg16_bn_lth --load_ff --resume > VGG16-Cifar10-ft.log 2>&1 &

# (3) GA
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_cifar10/GA --unlearn GA --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --unlearn_epochs 4 --unlearn_lr 0.0001 --arch vgg16_bn_lth --load_ff --resume > VGG16-Cifar10-ga.log 2>&1 &

# (4) FF
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_cifar10/FF --unlearn fisher --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --unlearn_epochs 100 --unlearn_lr 0.1 --arch vgg16_bn_lth --load_ff --resume --alpha 16.5 > VGG16-Cifar10-ff.log 2>&1 &

# (5) IU
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_cifar10/IU --unlearn wfisher --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --unlearn_epochs 100 --unlearn_lr 0.1 --arch vgg16_bn_lth --load_ff --resume --alpha 1 > VGG16-Cifar10-iu.log 2>&1 &

# (6) FT_prune
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_cifar10/FT_prune --unlearn FT_prune --class_to_replace 0,1,3,5,6 --num_indexes_to_replace 2250 --unlearn_epochs 20 --unlearn_lr 0.00001 --arch vgg16_bn_lth --load_ff --resume > VGG16-Cifar10-ft-prune.log 2>&1 &

# 1.2.2 vgg16_bn_lth Cifar100
# (0) backbone_ft
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_cifar100/finetune_backbone --unlearn retrain --class_to_replace 0 --num_indexes_to_replace 1 --unlearn_epochs 100 --unlearn_lr 0.1 --arch vgg16_bn_lth  --dataset cifar100 > VGG16-Cifar100-bbft.log 2>&1 &

# (1) retrain
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_cifar100/retrain --unlearn retrain --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --unlearn_epochs 100 --unlearn_lr 0.1 --arch vgg16_bn_lth  --dataset cifar100 > VGG16-Cifar100-retrain.log 2>&1 &

# (2) FT
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_cifar100/FT --unlearn FT --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --unlearn_epochs 100 --unlearn_lr 0.1 --arch vgg16_bn_lth  --dataset cifar100 --load_ff --resume > VGG16-Cifar100-ft.log 2>&1 &

# (3) GA
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_cifar100/GA --unlearn GA --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --unlearn_epochs 4 --unlearn_lr 0.001 --arch vgg16_bn_lth  --dataset cifar100 --load_ff --resume > VGG16-Cifar100-ga.log 2>&1 &

# (4) FF
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_cifar100/FF --unlearn fisher --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --unlearn_epochs 100 --unlearn_lr 0.1 --arch vgg16_bn_lth  --dataset cifar100 --load_ff --resume --alpha 16.5 > VGG16-Cifar100-ff.log 2>&1 &

# (5) IU
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_cifar100/IU --unlearn wfisher --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --unlearn_epochs 100 --unlearn_lr 0.1 --arch vgg16_bn_lth  --dataset cifar100 --load_ff --resume --alpha 6 > VGG16-Cifar100-iu.log 2>&1 &

# (6) FT_prune
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_cifar100/FT_prune --unlearn FT_prune --class_to_replace 11,22,33,44,55 --num_indexes_to_replace 225 --unlearn_epochs 20 --unlearn_lr 0.000005 --arch vgg16_bn_lth  --dataset cifar100 --load_ff --resume > VGG16-Cifar100-ft-prune.log 2>&1 &

# 1.2.3 vgg16_bn_lth tiny image net
# (0) backbone_ft
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_tinyimg/finetune_backbone --unlearn retrain --class_to_replace 0 --num_indexes_to_replace 1 --unlearn_epochs 100 --unlearn_lr 0.1 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch vgg16_bn_lth > VGG16-tinyim-bbft.log 2>&1 &

# (1) retrain
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_tinyimg/retrain --unlearn retrain --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --unlearn_epochs 100 --unlearn_lr 0.1 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch vgg16_bn_lth > VGG16-tinyim-retrain.log 2>&1 &

# (2) FT
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_tinyimg/FT --unlearn FT --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --unlearn_epochs 100 --unlearn_lr 0.1 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch vgg16_bn_lth --load_ff --resume > VGG16-tinyim-ft.log 2>&1 &

# (3) GA
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_tinyimg/GA --unlearn GA --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --unlearn_epochs 4 --unlearn_lr 0.0001 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch vgg16_bn_lth --load_ff --resume > VGG16-tinyim-ga.log 2>&1 &

# (4) FF
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_tinyimg/FF --unlearn fisher --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --unlearn_epochs 100 --unlearn_lr 0.1 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch vgg16_bn_lth --load_ff --resume --alpha 16.5 > VGG16-tinyim-ff.log 2>&1 &

# (5) IU
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_tinyimg/IU --unlearn wfisher --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --unlearn_epochs 100 --unlearn_lr 0.1 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch vgg16_bn_lth --load_ff --resume --alpha 10 > VGG16-tinyim-iu.log 2>&1 &

# (6) FT_prune
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_tinyimg/FT_prune --unlearn FT_prune --class_to_replace 1,51,101,151,198 --num_indexes_to_replace 250 --unlearn_epochs 7 --unlearn_lr 0.000004 --dataset TinyImagenet --data_dir ../data/tiny-imagenet-200 --arch vgg16_bn_lth --load_ff --resume > VGG16-tinyim-ft-prune.log 2>&1 &

# 1.2.4 vgg16_bn_lth fashionMNIST
# (0) backbone_ft
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_fmnist/finetune_backbone --unlearn retrain --class_to_replace 0 --num_indexes_to_replace 1 --unlearn_epochs 100 --unlearn_lr 0.1 --dataset fashionMNIST --arch vgg16_bn_lth > VGG16-fashionmn-bbft.log 2>&1 &

# (1) retrain
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_fmnist/retrain --unlearn retrain --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --unlearn_epochs 100 --unlearn_lr 0.1 --dataset fashionMNIST --arch vgg16_bn_lth > VGG16-fashionmn-retrain.log 2>&1 &

# (2) FT
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_fmnist/FT --unlearn FT --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --unlearn_epochs 50 --unlearn_lr 0.1 --dataset fashionMNIST --arch vgg16_bn_lth --load_ff --resume > VGG16-fashionmn-ft.log 2>&1 &

# (3) GA
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_fmnist/GA --unlearn GA --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --unlearn_epochs 4 --unlearn_lr 0.0001 --dataset fashionMNIST --arch vgg16_bn_lth --load_ff --resume > VGG16-fashionmn-ga.log 2>&1 &

# (4) FF
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_fmnist/FF --unlearn fisher --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --unlearn_epochs 100 --unlearn_lr 0.1 --dataset fashionMNIST --arch vgg16_bn_lth --load_ff --resume --alpha 16.5 > VGG16-fashionmn-ff.log 2>&1 &

# (5) IU
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_fmnist/IU --unlearn wfisher --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --unlearn_epochs 100 --unlearn_lr 0.1 --dataset fashionMNIST --arch vgg16_bn_lth --load_ff --resume --alpha 40 > VGG16-fashionmn-iu.log 2>&1 &

# (6) FT_prune
nohup python -u main_forget.py --save_dir $BASE_DIR/outputs/vgg16_fmnist/FT_prune --unlearn FT_prune --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --unlearn_epochs 10 --unlearn_lr 0.00001 --dataset fashionMNIST --arch vgg16_bn_lth --load_ff --resume > VGG16-fashionmn-ft-prune.log 2>&1 &
