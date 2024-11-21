import shutil

from configs import settings
import os
import numpy as np
import torch


def split_by_class(data, labels, num_classes=1000):
    """按类别划分数据集"""
    class_data = {i: [] for i in range(num_classes)}
    for i, label in enumerate(labels):
        class_data[label.item()].append(data[i])
    return class_data


def sample_class_balanced_data(class_data, split_ratio):
    """按比例从每个类别中均衡抽取样本"""
    D_0_data = []
    D_0_labels = []
    D_inc_data = []
    D_inc_labels = []

    for class_label, samples in class_data.items():
        num_samples = len(samples)
        split_idx = int(num_samples * split_ratio)
        if split_idx < 10:
            split_idx = num_samples // 2

        shuffled_indices = np.random.permutation(num_samples)

        # D_0 获取前半部分数据
        D_0_data.extend([samples[i] for i in shuffled_indices[:split_idx]])
        D_0_labels.extend([class_label] * split_idx)

        # D_inc 获取后半部分数据
        D_inc_data.extend([samples[i] for i in shuffled_indices[split_idx:]])
        D_inc_labels.extend([class_label] * (num_samples - split_idx))

    D_0_data = np.stack(D_0_data)
    D_0_labels = np.array(D_0_labels)
    D_inc_data = np.stack(D_inc_data)
    D_inc_labels = np.array(D_inc_labels)

    return D_0_data, D_0_labels, D_inc_data, D_inc_labels


def sample_replay_data(D_0_data, D_0_labels, replay_ratio, num_classes=1000):
    """从 D_0 中均衡抽取样本作为重放数据集 D_a"""
    class_data = split_by_class(D_0_data, D_0_labels, num_classes)
    D_a_data = []
    D_a_labels = []

    for class_label, samples in class_data.items():
        num_samples = len(samples)
        num_replay_samples = int(num_samples * replay_ratio)
        if num_replay_samples < 10:
            num_replay_samples = num_samples

        replay_indices = np.random.choice(num_samples, num_replay_samples, replace=False)

        D_a_data.extend([samples[i] for i in replay_indices])
        D_a_labels.extend([class_label] * num_replay_samples)

    D_a_data = np.stack(D_a_data)
    D_a_labels = np.array(D_a_labels)

    return D_a_data, D_a_labels

def split(dataset_name, case, train_dataset=None, test_dataset=None, num_classes=1000):
    rawcase = None
    train_data_path = settings.get_dataset_path(dataset_name, rawcase, "train_data")
    train_label_path = settings.get_dataset_path(dataset_name, rawcase, "train_label")
    test_data_path = settings.get_dataset_path(dataset_name, rawcase, "test_data")
    test_label_path = settings.get_dataset_path(dataset_name, rawcase, "test_label")
    aux_data_path = settings.get_dataset_path(dataset_name, rawcase, "aux_data")
    aux__label_path = settings.get_dataset_path(dataset_name, rawcase, "aux_label")
    inc_data_path = settings.get_dataset_path(dataset_name, rawcase, "inc_data")
    inc_label_path = settings.get_dataset_path(dataset_name, rawcase, "inc_label")
    train_0_data_path = settings.get_dataset_path(dataset_name, rawcase, "train_0_data")
    train_0_label_path = settings.get_dataset_path(dataset_name, rawcase, "train_0_label")

    resplit = False if os.path.exists(train_0_label_path) else True

    if resplit:
        train_data, train_labels = zip(*train_dataset)
        train_data = torch.stack(train_data)
        train_labels = torch.tensor(train_labels)

        test_data, test_labels = zip(*test_dataset)
        test_data = torch.stack(test_data)
        test_labels = torch.tensor(test_labels)

        # 按类别划分训练数据
        class_data = split_by_class(train_data, train_labels, num_classes)

        # 构建类均衡的 D_0 和 D_inc_0
        D_0_data, D_0_labels, D_inc_data, D_inc_labels = sample_class_balanced_data(
            class_data, split_ratio=0.5
        )

        # 构建重放数据集 D_a（从 D_0 中随机抽取 10% 的样本）
        D_a_data, D_a_labels = sample_replay_data(
            D_0_data, D_0_labels, replay_ratio=0.1, num_classes=num_classes
        )

        subdir = os.path.dirname(train_data_path)
        os.makedirs(subdir, exist_ok=True)

        np.save(aux_data_path, D_a_data)
        np.save(aux__label_path, D_a_labels)

        # 保存训练数据集
        np.save(train_data_path, train_data)
        np.save(train_label_path, train_labels)

        # 保存测试数据集
        np.save(test_data_path, test_data)
        np.save(test_label_path, test_labels)

        np.save(inc_data_path, D_inc_data)
        np.save(inc_label_path, D_inc_labels)


        # 保存初始数据集、初始增量数据集、重放数据集
        np.save(train_0_data_path, D_0_data)
        np.save(train_0_label_path, D_0_labels)

    # train_data_path_case = settings.get_dataset_path(dataset_name, rawcase, "train_data")
    # train_label_path_case = settings.get_dataset_path(dataset_name, rawcase, "train_label")
    test_data_path_case = settings.get_dataset_path(dataset_name, case, "test_data")
    test_label_path_case = settings.get_dataset_path(dataset_name, case, "test_label")
    # inc_data_path_case = settings.get_dataset_path(dataset_name, case, "inc_data")
    # inc_label_path_case = settings.get_dataset_path(dataset_name, case, "inc_label")
    aux_data_path_case = settings.get_dataset_path(dataset_name, case, "aux_data")
    aux__label_path_case = settings.get_dataset_path(dataset_name, case, "aux_label")
    train_0_data_path_case = settings.get_dataset_path(dataset_name, case, "train_data", step=0)
    train_0_label_path_case = settings.get_dataset_path(dataset_name, case, "train_label", step=0)

    subdir = os.path.dirname(train_0_data_path_case)
    os.makedirs(subdir, exist_ok=True)

    # shutil.copy(train_data_path, train_data_path_case)
    # shutil.copy(train_label_path, train_label_path_case)
    shutil.copy(test_data_path, test_data_path_case)
    shutil.copy(test_label_path, test_label_path_case)
    # shutil.copy(inc_data_path, inc_data_path_case)
    # shutil.copy(inc_label_path, inc_label_path_case)
    shutil.copy(aux_data_path, aux_data_path_case)
    shutil.copy(aux__label_path, aux__label_path_case)
    shutil.copy(train_0_data_path, train_0_data_path_case)
    shutil.copy(train_0_label_path, train_0_label_path_case)

    return np.load(inc_data_path), np.load(inc_label_path)


def split_data(dataset_name, train_dataset=None, test_dataset=None, num_classes=1000, split_ratio=0.5):
    rawcase = None
    train_data_path = settings.get_dataset_path(dataset_name, rawcase, "train_data")
    train_label_path = settings.get_dataset_path(dataset_name, rawcase, "train_label")
    test_data_path = settings.get_dataset_path(dataset_name, rawcase, "test_data")
    test_label_path = settings.get_dataset_path(dataset_name, rawcase, "test_label")
    aux_data_path = settings.get_dataset_path(dataset_name, rawcase, "aux_data")
    aux__label_path = settings.get_dataset_path(dataset_name, rawcase, "aux_label")
    inc_data_path = settings.get_dataset_path(dataset_name, rawcase, "inc_data")
    inc_label_path = settings.get_dataset_path(dataset_name, rawcase, "inc_label")
    train_0_data_path = settings.get_dataset_path(dataset_name, rawcase, "pretrain_data")
    train_0_label_path = settings.get_dataset_path(dataset_name, rawcase, "pretrain_label")

    resplit = False if os.path.exists(train_0_label_path) else True

    if resplit:
        train_data, train_labels = zip(*train_dataset)
        train_data = torch.stack(train_data)
        train_labels = torch.tensor(train_labels)

        test_data, test_labels = zip(*test_dataset)
        test_data = torch.stack(test_data)
        test_labels = torch.tensor(test_labels)

        # 按类别划分训练数据
        class_data = split_by_class(train_data, train_labels, num_classes)

        # 构建类均衡的 D_0 和 D_inc_0
        D_0_data, D_0_labels, D_inc_data, D_inc_labels = sample_class_balanced_data(
            class_data, split_ratio=split_ratio
        )

        # 构建重放数据集 D_a（从 D_0 中随机抽取 10% 的样本）
        D_a_data, D_a_labels = sample_replay_data(
            D_0_data, D_0_labels, replay_ratio=0.1, num_classes=num_classes
        )

        subdir = os.path.dirname(train_data_path)
        os.makedirs(subdir, exist_ok=True)

        np.save(aux_data_path, D_a_data)
        np.save(aux__label_path, D_a_labels)

        # 保存训练数据集
        np.save(train_data_path, train_data)
        np.save(train_label_path, train_labels)

        # 保存测试数据集
        np.save(test_data_path, test_data)
        np.save(test_label_path, test_labels)

        np.save(inc_data_path, D_inc_data)
        np.save(inc_label_path, D_inc_labels)


        # 保存初始数据集、初始增量数据集、重放数据集
        np.save(train_0_data_path, D_0_data)
        np.save(train_0_label_path, D_0_labels)

    return np.load(inc_data_path), np.load(inc_label_path)

