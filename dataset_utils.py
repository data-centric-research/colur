import torch
import torchvision
import torchvision.transforms as transforms
import random
from torch.utils.data import Subset
import pandas as pd
from tabulate import tabulate
from PIL import Image
from dataset_loaders import *


class DatasetUtils:
    def __init__(self, dataset_name, dataset_paths, num_classes_dict):
        self.dataset_name = dataset_name
        self.dataset_paths = dataset_paths
        self.num_classes = num_classes_dict[dataset_name]
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_dataset()

    def get_dataset(self):
        if self.dataset_name == "cifar-10":
            train_loader, val_loader, test_loader = cifar10_dataloaders()
            train_dataset = train_loader.dataset
            val_dataset = val_loader.dataset
            test_dataset = test_loader.dataset

        elif self.dataset_name == "cifar-100":
            train_loader, val_loader, test_loader = cifar100_dataloaders()
            train_dataset = train_loader.dataset
            val_dataset = val_loader.dataset
            test_dataset = test_loader.dataset

        elif self.dataset_name == "flowers-102":
            train_loader, val_loader, test_loader = Flowers102_dataloaders()
            train_dataset = train_loader.dataset
            val_dataset = val_loader.dataset
            test_dataset = test_loader.dataset

        elif self.dataset_name == "tiny-imagenet-200":
            train_loader, val_loader, test_loader = tinyImageNet_dataloaders()
            train_dataset = train_loader.dataset
            val_dataset = val_loader.dataset
            test_dataset = test_loader.dataset

        else:
            raise ValueError("Unsupported dataset: " + self.dataset_name)

        return train_dataset, val_dataset, test_dataset

    def flatten_class_list(self, class_list):
        """
        扁平化类标签列表，处理嵌套列表。
        """
        flattened = []
        for item in class_list:
            if isinstance(item, list):
                flattened.extend(self.flatten_class_list(item))
            else:
                flattened.append(int(item))
        return flattened

    def remove_fraction_of_selected_classes(
        self, dataset, selected_classes, remove_fraction=0.5, remove_all=False
    ):
        """
        删除选定类别的指定比例样本或全部样本。
        """
        selected_classes = self.flatten_class_list(selected_classes)
        class_indices = {i: [] for i in selected_classes}

        for idx, (_, label) in enumerate(dataset):
            if label in selected_classes:
                class_indices[label].append(idx)

        removed_indices = []
        if remove_all:
            # 如果 remove_all 为 True，移除所有指定类别的样本
            for label in selected_classes:
                removed_indices.extend(class_indices[label])
        else:
            # 按照 remove_fraction 移除样本
            for label in selected_classes:
                indices = class_indices[label]
                removed_indices.extend(
                    random.sample(indices, int(len(indices) * remove_fraction))
                )

        remaining_indices = list(set(range(len(dataset))) - set(removed_indices))
        return Subset(dataset, remaining_indices)

    def add_noise_to_selected_classes(
        self, dataset, selected_classes, noise_fraction=0.8, noise_type="gaussian"
    ):
        """
        为选定类别添加噪声。
        """
        selected_classes = self.flatten_class_list(selected_classes)
        noisy_data = []
        noisy_labels = []

        for image, label in dataset:
            if label in selected_classes and random.random() < noise_fraction:
                if noise_type == "gaussian":
                    noise = torch.randn_like(image) * 0.1
                    image = torch.clamp(image + noise, -1, 1)
                elif noise_type == "salt_pepper":
                    image = self.add_salt_and_pepper_noise(image)
                elif noise_type == "motion_blur":
                    image = self.add_motion_blur_noise(image)
                elif noise_type == "shear":
                    image = transforms.RandomAffine(degrees=0, shear=20)(image)

            noisy_data.append(image)
            noisy_labels.append(label)

        return list(zip(noisy_data, noisy_labels))

    def add_salt_and_pepper_noise(self, image, amount=0.05, salt_vs_pepper=0.5):
        """
        添加椒盐噪声。
        """
        noisy_image = image.clone()
        num_salt = int(amount * image.numel() * salt_vs_pepper)
        num_pepper = int(amount * image.numel() * (1.0 - salt_vs_pepper))

        coords_salt = [torch.randint(0, i, (num_salt,)) for i in image.shape]
        coords_pepper = [torch.randint(0, i, (num_pepper,)) for i in image.shape]

        noisy_image[coords_salt] = 1
        noisy_image[coords_pepper] = 0

        return noisy_image

    def add_motion_blur_noise(self, image, kernel_size=5):
        """
        添加运动模糊噪声。
        """
        kernel_motion_blur = torch.zeros((kernel_size, kernel_size))
        kernel_motion_blur[int((kernel_size - 1) / 2), :] = torch.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size
        image = image.unsqueeze(0)
        image_blurred = torch.nn.functional.conv2d(
            image,
            kernel_motion_blur.unsqueeze(0).unsqueeze(0),
            padding=kernel_size // 2,
        )
        return image_blurred.squeeze(0)

    def modify_dataset(
        self,
        dataset,
        selected_classes_remove,
        selected_classes_noise,
        remove_fraction=0.5,
        noise_fraction=0.1,
        noise_type="gaussian",
        remove_all=False,
    ):
        """
        同时执行删除样本和注入噪声的操作。
        """
        dataset_after_removal = self.remove_fraction_of_selected_classes(
            dataset, selected_classes_remove, remove_fraction, remove_all
        )

        return self.add_noise_to_selected_classes(
            dataset_after_removal, selected_classes_noise, noise_fraction, noise_type
        )

    def compute_statistics(self, dataset):
        """
        计算数据集的类别分布、像素均值和标准差，以及每个通道的均值和标准差。
        """
        class_counts = {}
        pixel_means, pixel_stds = [], []
        channel_means, channel_stds = torch.zeros(3), torch.zeros(3)

        for image, label in dataset:
            class_counts[label] = class_counts.get(label, 0) + 1
            pixel_means.append(image.mean().item())
            pixel_stds.append(image.std().item())
            channel_means += image.mean(dim=(1, 2))
            channel_stds += image.std(dim=(1, 2))

        total_images = len(dataset)
        channel_means /= total_images
        channel_stds /= total_images

        return (
            class_counts,
            sum(pixel_means) / len(pixel_means),
            sum(pixel_stds) / len(pixel_stds),
            channel_means,
            channel_stds,
        )


if __name__ == "__main__":
    dataset_paths = {
        "cifar-10": "./data/cifar-10",
        "cifar-100": "./data/cifar-100",
        "animals-10": "./data/animals-10",
        "flowers-102": "./data/flowers-102",
        "tiny-imagenet-200": "./data/tiny-imagenet-200",
    }

    num_classes_dict = {
        "cifar-10": 10,
        "cifar-100": 100,
        "animals-10": 10,
        "flowers-102": 102,
        "tiny-imagenet-200": 200,
    }

    datasets_to_test = ["cifar-10", "cifar-100", "flowers-102", "tiny-imagenet-200"]

    for dataset_name in datasets_to_test:
        print(f"\nTesting dataset: {dataset_name}")

        # 初始化 DatasetUtils
        dataset_loader = DatasetUtils(
            dataset_name=dataset_name,
            dataset_paths=dataset_paths,
            num_classes_dict=num_classes_dict,
        )

        # 获取数据集
        train_dataset, val_dataset, test_dataset = dataset_loader.get_dataset()

        # 计算并显示原始数据集的统计信息
        (
            original_class_counts,
            original_mean,
            original_std,
            original_channel_means,
            original_channel_stds,
        ) = dataset_loader.compute_statistics(train_dataset)

        # 修改数据集 - 移除部分类别的样本并注入噪声
        modified_dataset = dataset_loader.modify_dataset(
            dataset=train_dataset,
            selected_classes_remove=[0, 1, 2, 3, 4],
            selected_classes_noise=[5, 6, 7, 8, 9],
            remove_fraction=0.5,
            noise_fraction=0.5,
            noise_type="salt_pepper",
        )

        # 计算并显示修改后的数据集的统计信息
        (
            modified_class_counts,
            modified_mean,
            modified_std,
            modified_channel_means,
            modified_channel_stds,
        ) = dataset_loader.compute_statistics(modified_dataset)

        # 打印统计信息
        data = {
            "Metric": [
                "Class Distribution",
                "Pixel Mean",
                "Pixel Std",
                "Channel Means",
                "Channel Stds",
            ],
            "Original": [
                original_class_counts,
                original_mean,
                original_std,
                original_channel_means,
                original_channel_stds,
            ],
            "Modified": [
                modified_class_counts,
                modified_mean,
                modified_std,
                modified_channel_means,
                modified_channel_stds,
            ],
        }
        df = pd.DataFrame(data)
        print(tabulate(df, headers="keys", tablefmt="grid"))
