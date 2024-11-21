import numpy as np
import os
import argparse


def load_npy_files(data_path, labels_path):
    data = np.load(data_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    return data, labels


def validate_split(train_data, D_0_data, D_1_data, D_0_indices_path, D_1_indices_path):
    D_0_indices = np.load(D_0_indices_path, allow_pickle=True)
    D_1_indices = np.load(D_1_indices_path, allow_pickle=True)

    assert len(D_0_indices) + len(D_1_indices) == len(
        train_data
    ), "D_0 和 D_1 的索引总数不匹配原始数据集大小。"
    assert (
        len(set(D_0_indices).intersection(set(D_1_indices))) == 0
    ), "D_0 和 D_1 存在重叠样本。"

    D_0_from_train = [train_data[i] for i in D_0_indices]
    D_1_from_train = [train_data[i] for i in D_1_indices]

    assert np.array_equal(D_0_data, np.array(D_0_from_train)), "D_0 数据不匹配。"
    assert np.array_equal(D_1_data, np.array(D_1_from_train)), "D_1 数据不匹配。"
    print("D_0 和 D_1 的划分正确。")


def validate_noise(D_1_minus_labels, D_1_plus_labels, noise_ratio):
    num_noisy_samples = int(np.ceil(len(D_1_minus_labels) * noise_ratio))
    diff_labels = np.sum(D_1_minus_labels != D_1_plus_labels)

    assert (
        diff_labels == num_noisy_samples
    ), f"噪声比例不匹配，实际添加了 {diff_labels} 个噪声样本，应为 {num_noisy_samples}"
    print("噪声添加验证成功。")


def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="验证生成的 FLOWER-102 数据集的正确性。"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/flower-102/normal",
        help="原始 FLOWER-102 数据集的目录",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        default="./data/flower-102/gen/",
        help="生成数据集的保存目录",
    )
    parser.add_argument(
        "--noise_ratio", type=float, default=0.2, help="噪声比例（默认 0.2）"
    )

    args = parser.parse_args()

    # 加载原始训练数据集
    train_data_path = os.path.join(args.data_dir, "train_data.npy")
    train_labels_path = os.path.join(args.data_dir, "train_label.npy")
    train_data, train_labels = load_npy_files(train_data_path, train_labels_path)

    # 加载 D_0, D_1_minus 和 D_1_plus 数据集
    D_0_data, D_0_labels = load_npy_files(
        os.path.join(args.gen_dir, "D_0_data.npy"),
        os.path.join(args.gen_dir, "D_0_label.npy"),
    )
    D_1_minus_data, D_1_minus_labels = load_npy_files(
        os.path.join(args.gen_dir, "D_1_minus_data.npy"),
        os.path.join(args.gen_dir, "D_1_minus_label.npy"),
    )
    D_1_plus_data, D_1_plus_labels = load_npy_files(
        os.path.join(args.gen_dir, "D_1_plus_data.npy"),
        os.path.join(args.gen_dir, "D_1_plus_label.npy"),
    )

    # 验证 D_0 和 D_1 的划分
    validate_split(
        train_data,
        D_0_data,
        D_1_minus_data,
        os.path.join(args.gen_dir, "D_0_indices.npy"),
        os.path.join(args.gen_dir, "D_1_indices.npy"),
    )

    # 验证噪声添加
    validate_noise(D_1_minus_labels, D_1_plus_labels, args.noise_ratio)

    print("生成的数据集验证完成，一切正常。")


if __name__ == "__main__":
    main()
