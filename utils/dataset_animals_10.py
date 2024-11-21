import os
from PIL import Image
from torchvision.datasets import DatasetFolder
import torchvision.transforms as transforms
from torch.utils.data import random_split


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def get_animals_10_dataset(dataset_path, transform=None, split_ratio=0.8):
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    dataset = DatasetFolder(
        root=os.path.join(dataset_path, "raw-img"),
        loader=pil_loader,
        extensions=("jpg", "jpeg", "png"),
        transform=transform,
    )

    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset
