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
    Load dataset file and return it as a PyTorch tensor.
    :param subdir: Data directory
    :param dataset_name: Dataset name (cifar-10, cifar-100, food-101, pet-37, flower-102)
    :param file_name: Data file name
    :param is_data: Whether it is a data file (True for data file, False for label file)
    :return: Data in PyTorch tensor format
    """
    data = np.load(file_path)

    if is_data:
        data_tensor = torch.tensor(data, dtype=torch.float32)
    else:
        data_tensor = torch.tensor(data, dtype=torch.long)

    return data_tensor


def train_step(
    args,
    writer=None,
):
    """
    Train model according to the step
    :param step: The step to execute (0, 1, 2, ...)
    :param subdir: Path to the data subdirectory
    :param ckpt_subdir: Path to the model checkpoint subdirectory
    :param output_dir: Directory to save the model
    :param dataset_name: Type of dataset used (cifar-10 or cifar-100)
    :param load_model_path: Path to the specified model to load (optional)
    :param epochs: Number of training epochs
    :param batch_size: Batch size
    :param optimizer_type: Optimizer
    :param learning_rate: Learning rate
    """
    warnings.filterwarnings("ignore")

    dataset_name = args.dataset
    num_classes = get_num_of_classes(dataset_name)

    print(f"===== Executing step: {args.step} =====")
    print(f"Dataset name: {dataset_name}")
    print(
        f"Epochs: {args.num_epochs}, Batch Size: {args.batch_size}, Learning Rate: {args.learning_rate}"
    )

    model_name = args.model
    train_mode = args.train_mode

    case = settings.get_case(args.noise_ratio, args.noise_type)

    uni_name = args.uni_name

    model_suffix = "restore"

    test_data = load_dataset(settings.get_dataset_path(dataset_name, None, "test_data"))
    test_labels = load_dataset(
        settings.get_dataset_path(dataset_name, None, "test_label"), is_data=False
    )

    if train_mode == "pretrain" or train_mode == "train":
        model_p0_path = settings.get_ckpt_path(
            dataset_name, "pretrain", model_name, "pretrain"
        )

        if uni_name is None:
            train_data = np.load(
                settings.get_dataset_path(dataset_name, None, f"{train_mode}_data")
            )
            train_labels = np.load(
                settings.get_dataset_path(dataset_name, None, f"{train_mode}_label")
            )

            load_pretrained = True
            model_p0 = load_custom_model(
                model_name, num_classes, load_pretrained=load_pretrained
            )
            model_p0 = ClassifierWrapper(model_p0, num_classes)

            print(f"Start pretrain on ({dataset_name})...")

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
            print(f"Pretrained model saved to  {model_p0_path}")
    else:
        if train_mode == "retrain":  # Step 1: Train M_1 on D_1+ (noisy dataset)
            train_data = load_dataset(
                settings.get_dataset_path(dataset_name, case, "train_clean_data")
            )
            train_labels = load_dataset(
                settings.get_dataset_path(dataset_name, case, "train_clean_label"),
                is_data=False,
            )

            prev_model_path = settings.get_ckpt_path(
                dataset_name, "pretrain", model_name, "pretrain"
            )

            uni_name = train_mode
        elif train_mode == "finetune":  # Step 1: Train M_1 on D_1+ (noisy dataset)
            train_data = load_dataset(
                settings.get_dataset_path(dataset_name, case, "train_clean_data")
            )
            train_labels = load_dataset(
                settings.get_dataset_path(dataset_name, case, "train_clean_label"),
                is_data=False,
            )

            prev_model_path = settings.get_ckpt_path(
                dataset_name, case, model_name, "inc_train"
            )

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

            prev_model_path = settings.get_ckpt_path(
                dataset_name, "pretrain", model_name, "pretrain"
            )

            uni_name = None
            model_suffix = train_mode

        if not os.path.exists(prev_model_path):
            raise FileNotFoundError(
                f"Model file {prev_model_path} not found。Do pretrain first."
            )

        model_tr = load_custom_model(
            model_name=model_name, num_classes=num_classes, load_pretrained=False
        )
        model_tr = ClassifierWrapper(model_tr, num_classes)
        model_tr.load_state_dict(torch.load(prev_model_path))
        print(f"Start training {train_mode} on ({dataset_name})...")

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
        print(f"{train_mode} saved to {model_tr_path}")


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
    main()
