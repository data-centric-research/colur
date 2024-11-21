import os
import shutil
import warnings
import numpy as np
from args_paser import parse_args

import torch
from core_model.custom_model import ClassifierWrapper, load_custom_model
from configs import settings
from train_test_utils import train_model

def get_num_of_classes(dataset_name):
    # 根据 dataset_name 设置分类类别数
    if dataset_name == "cifar-10":
        num_classes = 10
    elif dataset_name == "pet-37":
        num_classes = 37
    elif dataset_name == "cifar-100":
        num_classes = 100
    elif dataset_name == "food-101":
        num_classes = 101
    elif dataset_name == "flower-102":
        num_classes = 102
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

    return num_classes


def load_dataset(file_path, is_data=True):
    """
    加载数据集文件并返回 PyTorch 张量。
    :param subdir: 数据目录
    :param dataset_name: 数据集名称 (cifar-10, cifar-100, food-101, pet-37, flower-102)
    :param file_name: 数据文件名
    :param is_data: 是否为数据文件（True 表示数据文件，False 表示标签文件）
    :return: PyTorch 张量格式的数据
    """
    data = np.load(file_path)

    if is_data:
        # 对于数据文件，转换为 float32 类型
        data_tensor = torch.tensor(data, dtype=torch.float32)
    else:
        # 对于标签文件，转换为 long 类型
        data_tensor = torch.tensor(data, dtype=torch.long)

    return data_tensor


def train_step(
    args,
    writer=None,
):
    """
    根据步骤训练模型
    :param step: 要执行的步骤（0, 1, 2, ...）
    :param subdir: 数据子目录路径
    :param ckpt_subdir: 模型检查点子目录路径
    :param output_dir: 模型保存目录
    :param dataset_name: 使用的数据集类型（cifar-10 或 cifar-100）
    :param load_model_path: 指定加载的模型路径（可选）
    :param epochs: 训练的轮数
    :param batch_size: 批次大小
    :optimizer_type: 优化器
    :param learning_rate: 学习率
    """
    warnings.filterwarnings("ignore")

    dataset_name = args.dataset
    num_classes = get_num_of_classes(dataset_name)

    # 打印当前执行的参数
    print(f"===== 执行步骤: {args.step} =====")
    print(f"数据集类型: {dataset_name}")
    print(
        f"Epochs: {args.num_epochs}, Batch Size: {args.batch_size}, Learning Rate: {args.learning_rate}"
    )

    model_name = args.model
    train_mode = args.train_mode
    
    case = settings.get_case(args.noise_ratio, args.noise_type)
    
    uni_name = args.uni_name

    model_suffix = "restore"

    test_data = load_dataset(
        settings.get_dataset_path(dataset_name, None, "test_data")
    )
    test_labels = load_dataset(
        settings.get_dataset_path(dataset_name, None, "test_label"), is_data=False
    )

    if train_mode == "pretrain" or train_mode == "train":  # 基于$D_0$数据集和原始的resnet网络训练一个模型 M_p0
        # model_p0_path = settings.get_pretrain_ckpt_path(
        #     dataset_name, case, model_name, "worker_restore", step=step
        # )
        model_p0_path = settings.get_ckpt_path(
            dataset_name, "pretrain", model_name, "pretrain")

        if uni_name is None:
            train_data = np.load(
                settings.get_dataset_path(dataset_name, None, f"{train_mode}_data")
            )
            train_labels = np.load(
                settings.get_dataset_path(dataset_name, None, f"{train_mode}_label")
            )

            # 打印用于训练的模型和数据
            print("用于训练的数据: pretrain_data.npy 和 pretrain_label.npy")
            print("用于训练的模型: ResNet18 初始化")

            load_pretrained = True  #False if dataset_name == "cifar-10" or dataset_name == "cifar-100" else True
            model_p0 = load_custom_model(model_name, num_classes, load_pretrained=load_pretrained)
            model_p0 = ClassifierWrapper(model_p0, num_classes)

            print(f"开始训练 Pretrain on ({dataset_name})...")

            model_p0 = train_model(
                model_p0,
                num_classes,
                train_data,
                train_labels,
                test_data,
                test_labels,
                epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                data_aug=args.data_aug,
                dataset_name=args.dataset,
                writer=writer,
            )
            subdir = os.path.dirname(model_p0_path)
            os.makedirs(subdir, exist_ok=True)
            torch.save(model_p0.state_dict(), model_p0_path)
            print(f"Pretrain 训练完毕并保存至 {model_p0_path}")
        # else:
        #     copy_model_p0_path = settings.get_ckpt_path(
        #         dataset_name,
        #         case,
        #         model_name,
        #         "_pretrain",
        #         unique_name=uni_name,
        #     )
        #     if os.path.exists(model_p0_path):
        #         subdir = os.path.dirname(copy_model_p0_path)
        #         os.makedirs(subdir, exist_ok=True)
        #         shutil.copy(model_p0_path, copy_model_p0_path)
        #         print(f"Copy {model_p0_path} to {copy_model_p0_path}")
        #     else:
        #         raise FileNotFoundError(model_p0_path)
    else:
        if train_mode == "retrain":  # Step 1: Train M_1 on D_1+ (noisy dataset)
            train_data = load_dataset(
                settings.get_dataset_path(dataset_name, case, "train_clean_data")
            )
            train_labels = load_dataset(
                settings.get_dataset_path(dataset_name, case, "train_clean_label"),
                is_data=False,
            )

            # 打印用于训练的模型和数据
            print("用于训练的数据: train_clean_data 和 train_clean_label")

            prev_model_path = settings.get_ckpt_path(
                dataset_name, "pretrain", model_name, "pretrain")

            uni_name = train_mode
        elif train_mode == "finetune":  # Step 1: Train M_1 on D_1+ (noisy dataset)
            train_data = load_dataset(
                settings.get_dataset_path(dataset_name, case, "train_clean_data")
            )
            train_labels = load_dataset(
                settings.get_dataset_path(dataset_name, case, "train_clean_label"),
                is_data=False,
            )

            # 打印用于训练的模型和数据
            print("用于训练的数据: train_clean_data 和 train_clean_label")

            prev_model_path = settings.get_ckpt_path(
                dataset_name, case, model_name, "inc_train")

            uni_name = train_mode
        elif train_mode == "inc_train":  # Step 1: Train M_1 on D_1+ (noisy dataset)
            train_clean_data = load_dataset(
                settings.get_dataset_path(dataset_name, case, "train_clean_data")
            )
            train_clean_labels = load_dataset(
                settings.get_dataset_path(dataset_name, case, "train_clean_label"),
                is_data=False,
            )
            train_noisy_data = load_dataset(
                settings.get_dataset_path(dataset_name, case, "train_noisy_data")
            )
            train_noisy_labels = load_dataset(
                settings.get_dataset_path(dataset_name, case, "train_noisy_label"),
                is_data=False,
            )
            train_data = torch.concatenate([train_clean_data, train_noisy_data])
            train_labels = torch.concatenate([train_clean_labels, train_noisy_labels])

            # 打印用于训练的模型和数据
            print("用于训练的数据: train_inc_data 和 train_inc_label")

            prev_model_path = settings.get_ckpt_path(
                dataset_name, "pretrain", model_name, "pretrain")

            uni_name = None
            model_suffix = train_mode

        if not os.path.exists(prev_model_path):
            raise FileNotFoundError(
                f"模型文件 {prev_model_path} 未找到。请先训练 pretrain。"
            )

        model_tr= load_custom_model(
            model_name=model_name, num_classes=num_classes, load_pretrained=False
        )
        model_tr = ClassifierWrapper(model_tr, num_classes)
        model_tr.load_state_dict(torch.load(prev_model_path))
        print(f"开始训练 {train_mode} on ({dataset_name})...")

        if len(train_data) == 0:
            print(f"len of train data is 0")

        model_tr = train_model(
            model_tr,
            num_classes,
            train_data,
            train_labels,
            test_data,
            test_labels,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            data_aug=args.data_aug,
            dataset_name=args.dataset,
            writer=writer,
        )

        model_tr_path = settings.get_ckpt_path(
            dataset_name, case, model_name, model_suffix, unique_name=uni_name
        )
        subdir = os.path.dirname(model_tr_path)
        os.makedirs(subdir, exist_ok=True)
        torch.save(model_tr.state_dict(), model_tr_path)
        print(f"{train_mode} 训练完毕并保存至 {model_tr_path}")


def main():
    args = parse_args()

    writer = None
    if args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir="runs/experiment")

    train_step(
        args,
        writer=writer,
    )

    if writer:
        writer.close()


if __name__ == "__main__":
    
    '''
    # 以CIFAR-10为例，步骤如下：
    
    # 1. 使用 D_0 训练 M_0
    python run_experiment_cvpr.py --dataset cifar-10 --model resnet18 --data_dir data/cifar-10/gen --step 0 --num_epochs 50 --batch_size 64 --learning_rate 0.001 --gpu 7
    
    # 2. 使用 D_1+ 训练 M_1
    python run_experiment_cvpr.py --dataset cifar-10 --model resnet18 --data_dir data/cifar-10/gen --step 1 --num_epochs 50 --batch_size 64 --learning_rate 0.001 --gpu 7
    
    # 3. 使用 D_1- 训练（修复） M_1
    python run_experiment_cvpr.py --dataset cifar-10 --model resnet18 --data_dir data/cifar-10/gen --step 2 --num_epochs 50 --batch_size 64 --learning_rate 0.001 --gpu 7
    
    '''
    
    main()
