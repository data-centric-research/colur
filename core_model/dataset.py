import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from configs import settings
import torch


class BaseTensorDataset(Dataset):

    def __init__(self, data, labels, transforms=None, output_index=False):
        self.data = data
        self.labels = labels
        self.transforms = transforms
        self.output_index = output_index

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if self.transforms is not None:
            data = self.transforms(data)
        if self.output_index:
            return data, self.labels[index], index

        return data, self.labels[index]


def normalize_dataset(dataset, channel_first=True, mean=None, std=None):
    shape = dataset.shape
    channel_idx = np.where(np.array(shape)[1:] == 3)[0]

    # modify shape to [N, C, H, W]
    axes = None
    if channel_first:
        if channel_idx == 2:
            axes = [0, 2, 1, 3]
        elif channel_idx == 3:
            axes = [0, 3, 1, 2]
    else:
        if channel_idx == 1:
            axes = [0, 2, 3, 1]
        elif channel_idx == 2:
            axes = [0, 1, 3, 2]

    if axes is not None:
        dataset = np.transpose(dataset, axes)

    # The following code will cause a bug because the current datasets have already been normalized.
    # At this point, some values may exceed 1, e.g., 1.xxx, but they are not pixel values.
    # This would incorrectly trigger the condition, causing all values to be divided by 255 again.
    # To avoid issues, simply comment out this line.

    # normalize
    # if (dataset[0] > 1).any():
    #     dataset = dataset / 255.
    # gaussian normalize

    if mean is not None:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        dataset = (dataset - mean) / std

    return dataset


def transform_data(data):
    shape = data.shape[1:]
    data_mod = random_crop(data, shape)
    data_mod = random_horiz_flip(data_mod)
    return data_mod


class MixupDataset(Dataset):
    def __init__(
        self,
        data_pair,
        label_pair,
        mixup_alpha=0.2,
        transforms=None,
        mean=None,
        std=None,
        first_max=True,
    ):
        # modify shape to [N, H, W, C]
        self.data_first = data_pair[0]
        self.data_second = data_pair[1]
        self.data_first = normalize_dataset(self.data_first, mean=mean, std=std)
        self.data_second = normalize_dataset(self.data_second, mean=mean, std=std)
        self.label_first = label_pair[0]
        self.label_second = label_pair[1]
        self.mixup_alpha = mixup_alpha
        self.transforms = transforms
        self.first_max = first_max

    def __len__(self):
        return len(self.label_first)

    def __getitem__(self, index):
        if self.mixup_alpha >= 0:
            lbd = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            if self.first_max and lbd < 0.5:
                lbd = 1 - lbd
        else:
            lbd = np.random.beta(-self.mixup_alpha, -self.mixup_alpha)
            if self.first_max and lbd > 0.5:
                lbd = 1 - lbd

        data_first = self.data_first[index]
        label_first = self.label_first[index]
        rnd_idx = np.random.randint(len(self.data_second))
        data_rnd_ax = self.data_second[rnd_idx]
        label_rnd_ax = self.label_second[rnd_idx]

        if self.transforms is not None:
            data_first = self.transforms(data_first)
            data_rnd_ax = self.transforms(data_rnd_ax)

        mixed_data = lbd * data_first + (1 - lbd) * data_rnd_ax
        mixed_labels = lbd * label_first + (1 - lbd) * label_rnd_ax
        return mixed_data, mixed_labels


class NormalizeDataset(BaseTensorDataset):
    def __init__(
        self,
        data,
        labels,
        transforms=None,
        output_index=False,
        channel_first=True,
        mean=None,
        std=None,
        device=None,
    ):
        super().__init__(data, labels, transforms, output_index)
        self.data = normalize_dataset(data, channel_first, mean, std)
        if device is not None:
            self.data = torch.as_tensor(data, device=device)


def get_dataset_loader(
    dataset_name,
    loader_name,
    case,
    step=None,
    mean=None,
    std=None,
    batch_size=64,
    num_classes=0,
    drop_last=False,
    shuffle=False,
    onehot_enc=False,
    transforms=None,
    output_index=False,
    data_name=None,
    label_name=None,
    device=None,
    num_workers=0,
):
    """
    Load the corresponding dataset based on loader_name: supports incremental training (inc), auxiliary data (aux), test data (test), and D0 dataset (train).
    """
    if not isinstance(loader_name, (list, tuple)):
        loader_name = [loader_name]

    data = []
    labels = []
    for ld_name in loader_name:
        if data_name is None:
            data_name = f"{ld_name}_data"
        data_path = settings.get_dataset_path(dataset_name, case, data_name, step)
        if label_name is None:
            label_name = f"{ld_name}_label"
        label_path = settings.get_dataset_path(dataset_name, case, label_name, step)

        print(f"Loading {data_path}")

        data.append(np.load(data_path))
        label = np.load(label_path)
        labels.append(label.astype(np.int64))

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    if onehot_enc:  # train label change to onehot for teacher model
        labels = np.eye(num_classes)[labels]

    # Build custom dataset
    dataset = NormalizeDataset(
        data,
        labels,
        transforms=transforms,
        output_index=output_index,
        mean=mean,
        std=std,
    )
    # dataset = BaseTensorDataset(data, labels, transforms=transforms)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    # if dataset_name == "pet-37":
    #     data_loader = DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         drop_last=drop_last,
    #         shuffle=shuffle,
    #         num_workers=64,
    #         pin_memory=True
    #     )

    return data, labels, data_loader


def random_crop(img, img_size, padding=4):
    img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), "constant")
    h, w = img.shape[1:]

    new_h, new_w = img_size
    start_x = np.random.randint(0, w - new_w)
    start_y = np.random.randint(0, h - new_h)

    crop_img = img[start_y : start_y + new_h, start_x : start_x + new_w]
    return crop_img


def random_horiz_flip(img):
    if random.random() > 0.5:
        img = np.fliplr(img)
    return img
