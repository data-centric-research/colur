import torchvision
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms


# Helper function to get the flowers dataset for train or val split
def get_flowers_train_or_val(root_dir: str, split: str):
    return torchvision.datasets.Flowers102(
        root=root_dir,
        download=True,
        split=split,
        transform=torchvision.transforms.Compose(
            [
                torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1.transforms(),
            ]
        ),
    )


def get_flowers_102_dataset(root_dir: str):
    flowers_train_dataset = torchvision.datasets.Flowers102(
        root=root_dir,
        split="test",
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3),
                torchvision.transforms.RandomAffine(degrees=30, shear=20),
                torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1.transforms(),
            ]
        ),
    )

    flowers_test_dataset = ConcatDataset(
        [
            get_flowers_train_or_val(root_dir, "train"),
            get_flowers_train_or_val(root_dir, "val"),
        ]
    )

    return flowers_train_dataset, flowers_test_dataset
