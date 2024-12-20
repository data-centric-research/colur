import json
import torch
import numpy as np
import os
import argparse
from torchvision import datasets, transforms
from configs import settings
from gen_dataset.split_dataset import split_data

conference_name = "cvpr"


def create_dataset_files(
    data_dir,
    gen_dir,
    dataset_name="cifar-10",
    noise_type="symmetric",
    noise_ratio=0.25,
    split_ratio=0.4,
):
    rng = np.random.default_rng(42)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.491, 0.482, 0.446], [0.247, 0.243, 0.261]),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    dataset_name = "cifar-10"
    num_classes = 10
    D_inc_data, D_inc_labels = split_data(
        dataset_name, train_dataset, test_dataset, num_classes, split_ratio
    )

    num_noisy_samples = int(len(D_inc_labels) * noise_ratio)
    noisy_indices = rng.choice(len(D_inc_labels), num_noisy_samples, replace=False)
    noisy_sel = np.zeros(len(D_inc_labels), dtype=np.bool_)
    noisy_sel[noisy_indices] = True

    D_noisy_data = D_inc_data[noisy_sel]
    D_noisy_true_labels = D_inc_labels[noisy_sel]

    D_normal_data = D_inc_data[~noisy_sel]
    D_normal_labels = D_inc_labels[~noisy_sel]

    asymmetric_mapping = {
        0: 5,
        1: 6,
        2: 7,
        3: 8,
        4: 9,
        5: 0,
        6: 1,
        7: 2,
        8: 3,
        9: 4,
    }

    if noise_type == "symmetric":
        D_noisy_labels = rng.choice(num_classes, num_noisy_samples, replace=True)
    elif noise_type == "asymmetric":
        D_noisy_labels = np.array(
            [
                asymmetric_mapping[label] if label in asymmetric_mapping else label
                for label in D_noisy_true_labels
            ]
        )
    else:
        raise ValueError("Invalid noise type.")

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
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(
        description="Generate CIFAR-10 experimental datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(settings.root_dir, "data/cifar-10/normal"),
        help="Directory of the original CIFAR-10 dataset",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        default=os.path.join(settings.root_dir, "data/cifar-10/gen/"),
        help="Directory to save the generated datasets",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["cifar-10"],
        default="cifar-10",
        help="Dataset only supports: 'cifar-10'",
    )
    parser.add_argument(
        "--split_ratio", type=float, default=0.4, help="split ratio(default 0.4)"
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["symmetric", "asymmetric"],
        default="symmetric",
        help="Label noise type: 'symmetric''",
    )
    parser.add_argument(
        "--noise_ratio", type=float, default=0.25, help="noise ratio(default 0.25)"
    )

    args = parser.parse_args()

    create_dataset_files(
        data_dir=args.data_dir,
        gen_dir=args.gen_dir,
        dataset_name=args.dataset_name,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        split_ratio=args.split_ratio,
    )


if __name__ == "__main__":
    main()
