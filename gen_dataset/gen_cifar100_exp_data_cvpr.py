import torch
import numpy as np
import os
import argparse
from torchvision import datasets, transforms
import json

from configs import settings
from gen_dataset.split_dataset import split_data


conference_name = "cvpr"


def load_classes_from_file(file_path):
    """read class list from file"""
    with open(file_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def load_cifar100_superclass_mapping(file_path):
    """Load the superclass and child class mapping of CIFAR-100 from a JSON file"""
    with open(file_path, "r") as f:
        cifar100_superclass_to_child = json.load(f)
    return cifar100_superclass_to_child


def build_asymmetric_mapping(superclass_mapping, classes, rng):
    """Build asymmetric label mapping to ensure labels are replaced with other classes within the same superclass"""
    child_to_superclass_mapping = {}

    # Build the reverse mapping from child class to superclass
    for superclass, child_classes in superclass_mapping.items():
        for child_class in child_classes:
            child_to_superclass_mapping[child_class] = (superclass, child_classes)

    # Build the asymmetric mapping table
    asymmetric_mapping = {}

    for class_name in classes:
        # Get the superclass of the category and all categories within that superclass
        if class_name in child_to_superclass_mapping:
            superclass, child_classes = child_to_superclass_mapping[class_name]
            # Randomly select a different category within the same superclass as the replacement
            available_classes = [c for c in child_classes if c != class_name]
            if available_classes:
                new_class = rng.choice(available_classes)
                asymmetric_mapping[class_name] = new_class
            else:
                asymmetric_mapping[class_name] = (
                    class_name# If there are no other categories, keep the original label unchanged
                )
    return asymmetric_mapping


def create_cifar100_npy_files(
    data_dir,
    gen_dir,
    noise_type="asymmetric",
    noise_ratio=0.25,
    split_ratio=0.6,
):
    rng = np.random.default_rng(87)

    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=data_transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=data_transform
    )

    case = settings.get_case(noise_ratio, noise_type)

    print("split training dataset...")
    dataset_name = "cifar-100"
    num_classes = 100
    D_inc_data, D_inc_labels = split_data(
        dataset_name, train_dataset, test_dataset, num_classes, split_ratio
    )

    cifar100_classes_file = os.path.join(
        settings.root_dir, "configs/classes/cifar_100_classes.txt"
    )
    cifar100_classes = load_classes_from_file(cifar100_classes_file)

    cifar100_mapping_file = os.path.join(
        settings.root_dir, "configs/classes/cifar_100_mapping.json"
    )
    cifar100_superclass_mapping = load_cifar100_superclass_mapping(
        cifar100_mapping_file
    )

    print("CIFAR-100 Classes:", cifar100_classes)

    if noise_type == "asymmetric":
        asymmetric_mapping = build_asymmetric_mapping(
            cifar100_superclass_mapping, cifar100_classes, rng
        )

    num_noisy_samples = int(len(D_inc_labels) * noise_ratio)
    noisy_indices = rng.choice(len(D_inc_labels), num_noisy_samples, replace=False)
    noisy_sel = np.zeros(len(D_inc_labels), dtype=np.bool_)
    noisy_sel[noisy_indices] = True

    D_noisy_data = D_inc_data[noisy_sel]
    D_noisy_true_labels = D_inc_labels[noisy_sel]
    D_normal_data = D_inc_data[~noisy_sel]
    D_normal_labels = D_inc_labels[~noisy_sel]

    D_noisy_labels = []
    for true_label in D_noisy_true_labels:
        original_class_name = cifar100_classes[true_label]
        if original_class_name in asymmetric_mapping:
            new_class_name = asymmetric_mapping[original_class_name]
            new_label = cifar100_classes.index(new_class_name)
        else:
            new_label = true_label
        D_noisy_labels.append(new_label)
    D_noisy_labels = np.array(D_noisy_labels)

    save_path = os.path.join(
        gen_dir, f"nr_{noise_ratio}_nt_{noise_type}_{conference_name}"
    )
    os.makedirs(save_path, exist_ok=True)

    D_1_minus_data_path = os.path.join(save_path, "train_clean_data.npy")
    D_1_minus_labels_path = os.path.join(save_path, "train_clean_label.npy")
    np.save(D_1_minus_data_path, np.array(D_normal_data))
    np.save(D_1_minus_labels_path, np.array(D_normal_labels))

    D_1_plus_data_path = os.path.join(save_path, "train_noisy_data.npy")
    D_1_plus_labels_path = os.path.join(save_path, "train_noisy_label.npy")
    D_1_plus_true_labels_path = os.path.join(save_path, "train_noisy_true_label.npy")
    np.save(D_1_plus_data_path, np.array(D_noisy_data))
    np.save(D_1_plus_labels_path, np.array(D_noisy_labels))
    np.save(D_1_plus_true_labels_path, np.array(D_noisy_true_labels))

    print("D_0, D_1_minus, and D_1_plus datasets have been generated and saved.")


def main():
    np.random.seed(31)
    torch.manual_seed(47)

    parser = argparse.ArgumentParser(
        description="Generate CIFAR-100 experimental datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/cifar-100/normal",
        help="Directory of the original CIFAR-100 dataset",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        default="./data/cifar-100/gen/",
        help="Directory to save the generated datasets",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["cifar-100"],
        default="cifar-100",
        help="Dataset only supports: 'cifar-100'",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["asymmetric"],
        default="asymmetric",
        help="Label noise type: currently only supports 'asymmetric'",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.6,
        help="Training set split ratio (default 0.6)",
    )
    parser.add_argument(
        "--noise_ratio", type=float, default=0.25, help="Noise ratio (default 0.25)"
    )

    args = parser.parse_args()

    create_cifar100_npy_files(
        data_dir=args.data_dir,
        gen_dir=args.gen_dir,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        split_ratio=args.split_ratio,
    )


if __name__ == "__main__":
    main()
