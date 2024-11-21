import os
import argparse

# from run_experiment import run_experiment


# 自定义检查函数
def check_positive(value):
    ivalue = float(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive float value")
    return ivalue


def check_fraction(value):
    fvalue = float(value)
    if not (0.0 <= fvalue <= 1.0):
        raise argparse.ArgumentTypeError(f"{value} is an invalid fraction (0.0 - 1.0)")
    return fvalue


# 解析 classes 参数（支持 0-9 形式）
def parse_class_range(value):
    if "-" in value:
        start, end = map(int, value.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(value)]


def parse_kwargs(kwargs):
    """
    将 --kwargs 中的 key=value 形式的输入解析为字典。
    """
    parsed_kwargs = {}
    if kwargs:
        for kwarg in kwargs:
            key, value = kwarg.split("=")
            parsed_kwargs[key] = (
                float(value) if "." in value else int(value)
            )  # 根据输入类型转换
    return parsed_kwargs


def make_arg_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Run experiments with different datasets, models, and conditions."
        )

    # 添加参数
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="cifar-10",
        choices=[
            "cifar-10",
            "cifar-100",
            "pet-37",
            "flower-102",
            "food-101",
        ],
        help="Dataset name, choose from: cifar-10, cifar-100, flower-102, tiny-imagenet-200, food-101",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Select in (cifar-resnet18, cifar-wideresnet40, cifar-resnet50, resnet18, resnet50, resnet101, vgg19, wideresnet50)",
    )

    # 在 parse_args 函数中添加 --pretrained 参数
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="If specified, use pretrained weights for the model",
    )

    parser.add_argument(
        "--no_spnorm",
        action="store_true",
        default=False,
        help="If specified, no spectral norm",
    )

    parser.add_argument(
        "--data_aug",
        action="store_true",
        default=False,
        help="If specified, do data augmentation",
    )

    parser.add_argument(
        "--condition",
        type=str,
        required=False,
        default="original_data",
        choices=["original_data", "remove_data", "noisy_data", "all_perturbations"],
        help="Condition for the experiment: original_data, remove_data, noisy_data, all_perturbations",
    )

    parser.add_argument(
        "--classes_remove",
        type=parse_class_range,
        nargs="+",
        required=False,
        help="List of classes to remove samples from, e.g., --classes_remove 0 1 2 3 4 or 0-4",
    )

    # 添加 remove_fraction 参数，用于指定删除样本的比例
    parser.add_argument(
        "--remove_fraction",
        type=check_fraction,  # 使用自定义函数
        default=0.5,
        help="Fraction of samples to remove from the selected classes, e.g., --remove_fraction 0.5 for 50%% removal (default: 0.5)",
    )

    parser.add_argument(
        "--classes_noise",
        type=parse_class_range,
        nargs="+",
        required=False,
        help="List of classes to add noise to, e.g., --classes_noise 5 6 7 8 9 or 5-9",
    )

    # 添加 noise_type 参数，用于指定噪声类型
    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["symmetric", "asymmetric"],
        default="symmetric",
        help="Noise type to use, e.g., symmetric or asymmetric",
    )

    parser.add_argument(
        "--balanced",
        default=False,
        action="store_true",
        help="是否使用类均衡的数据划分方式。如果不指定，则使用随机划分。",
    )

    parser.add_argument(
        "--train_aux",
        default=False,
        action="store_true",
        help="Training with auxiliary dataset",
    )

    parser.add_argument(
        "--tta_only",
        default=None,
        type=int,
        choices=[0, 1],
        help="",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Continual learning step",
    )

    parser.add_argument(
        "--train_mode",
        type=str,
        choices=["pretrain", "inc_train", "finetune", "retrain", "train"],
        help="Train mode",
    )

    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=0.2,
        help="Noise ratio",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="Specify the GPU(s) to use, e.g., --gpu 0,1 for multi-GPU or --gpu 0 for single GPU",
    )

    # 添加 batch_size 参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )

    # 添加 learning_rate 参数
    parser.add_argument(
        "--learning_rate",
        type=check_positive,  # 使用自定义函数
        default=0.001,
        help="Learning rate for the optimizer (default: 0.001)",
    )

    parser.add_argument(
        "--teacher_lr_scale",
        type=check_positive,  # 使用自定义函数
        default=0.2,
        help="Teacher learning rate scale",
    )

    # 添加 optimizer 参数
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adam"],
        default="adam",
        help="Optimizer for training weights",
    )

    # 添加 momentum 参数，用于指定 SGD 优化器的动量
    parser.add_argument(
        "--momentum",
        type=check_positive,
        default=0.9,
        help="Momentum for SGD optimizer (default: 0.9). Only used if optimizer is 'sgd'.",
    )

    # 添加 weight_decay 参数，用于指定权重衰减
    parser.add_argument(
        "--weight_decay",
        type=check_positive,
        default=5e-4,
        help="Weight decay for the optimizer (default: 0.0001).",
    )

    # 添加 num_epochs 参数
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of epochs to train the model (default: 200)",
    )

    parser.add_argument(
        "--ul_epochs",
        type=int,
        default=3,
        help="Number of unlearning epochs",
    )

    parser.add_argument(
        "--agree_epochs",
        type=int,
        default=2,
        help="Number of unlearning epochs (default: 3)",
    )

    # 添加 early_stopping_patience 参数
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Patience for early stopping (default: 10)",
    )

    # 添加 early_stopping_accuracy_threshold 参数
    parser.add_argument(
        "--early_stopping_accuracy_threshold",
        type=float,
        default=0.95,
        help="Accuracy threshold for early stopping (default: 0.95)",
    )

    # 添加早停开关参数
    parser.add_argument(
        "--use_early_stopping",
        action="store_true",
        help="Enable early stopping if specified, otherwise train for the full number of epochs",
    )

    # 修复训练的次数
    parser.add_argument(
        "--repair_iter_num",
        type=int,
        default=3,
        help="The number of iterations to train the model",
    )

    # 适应性训练的次数
    parser.add_argument(
        "--adapt_iter_num",
        type=int,
        default=2,
        help="The number of iterations to adapt the model",
    )

    parser.add_argument(
        "--adapt_epochs",
        type=int,
        default=1,
        help="The number of epochs to adapt the model",
    )

    parser.add_argument(
        "--lr_scale",
        type=float,
        default=0.5,
        help="Scale the working model lr",
    )

    parser.add_argument(
        "--ls_gamma",
        type=float,
        default=0.25,
        help="Label smoothing factor",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.75,
        help="Sharpen factor",
    )

    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.75,
        help="Mixup factor",
    )

    # 捕获其他 kwargs
    parser.add_argument(
        "--kwargs", nargs="*", help="Additional key=value arguments for hyperparameters"
    )

    # 捕获其他 kwargs
    parser.add_argument(
        "--model_suffix",
        type=str,
        default=None,
        help="Suffix to save model name",
    )

    parser.add_argument("--uni_name", type=str, default=None, help="Model unique name")

    parser.add_argument(
        "--use_tensorboard", action="store_true", help="Use TensorBoard for logging."
    )

    return parser


# 命令行解析
def parse_args():
    parser = make_arg_parser()
    # 返回解析的参数
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    # 设置 GPU 环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Using GPU(s): {args.gpu}")

    # 设置默认值为空列表，避免传递 None
    if args.classes_remove is None:
        args.classes_remove = []
    if args.classes_noise is None:
        args.classes_noise = []

    # 检查组合操作的条件
    if args.condition == "all_perturbations":
        if not args.classes_remove or not args.classes_noise:
            raise ValueError(
                "For 'all_perturbations' condition, both --classes_remove and --classes_noise must be provided."
            )

    # 解析 kwargs 参数
    kwargs = parse_kwargs(args.kwargs)

    # 打印用户输入的配置信息
    print(f"Running experiment with the following configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Condition: {args.condition}")
    if args.classes_remove:
        print(f"  Classes to Remove Samples From: {args.classes_remove}")
        print(f"  Remove Fraction: {args.remove_fraction}")
    if args.classes_noise:
        print(f"  Classes to Add Noise To: {args.classes_noise}")
        print(f"  Noise Type: {args.noise_type}")
        print(f"  Noise Fraction: {args.noise_fraction}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Optimizer: {args.optimizer}")
    if args.optimizer == "sgd":
        print(f"  Momentum: {args.momentum}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Number of Epochs: {args.num_epochs}")
    print(f"  Use Early Stopping: {args.use_early_stopping}")
    if args.use_early_stopping:
        print(f"  Early Stopping Patience: {args.early_stopping_patience}")
        print(
            f"  Early Stopping Accuracy Threshold: {args.early_stopping_accuracy_threshold}"
        )
    print(f"  Repair Iterations: {args.repair_iter_num}")
    print(f"  Adaptation Iterations: {args.adapt_iter_num}")

    print(f"Additional kwargs: {kwargs}")


if __name__ == "__main__":
    main()
