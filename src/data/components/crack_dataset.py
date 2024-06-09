from typing import List

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

channel_means = [0.598, 0.584, 0.565]
channel_stds = [0.104, 0.103, 0.103]


def _bextraction(img):
    img = img[0].numpy()
    img1 = img.astype(np.uint8)
    DIAMOND_KERNEL_5 = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    img2 = cv2.dilate(img1, DIAMOND_KERNEL_5).astype(img.dtype)
    img3 = img2 - img
    img3 = np.expand_dims(img3, axis=0)
    return torch.tensor(img3.copy())


class _ImgToTensor:
    def __call__(self, img: np.ndarray):
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(channel_means, channel_stds),
            ]
        )
        return tf(img)


class _MaskToTensor:
    def __call__(self, img: np.ndarray):
        return torch.from_numpy(img).float()


class CrackDataset(Dataset):
    """Dataset class for Crack datasets."""

    def __init__(
        self,
        x_set: List[str],
        y_set: List[str],
        img_size: int,
        is_train: bool = False,
        is_boundary: bool = False,
    ):

        self.x = x_set
        self.y = y_set
        self.img_size = (img_size, img_size)
        self.is_train = is_train
        self.is_boundary = is_boundary

        self.img_totensor = _ImgToTensor()
        self.mask_totensor = _MaskToTensor()

        self.transform = A.Compose(
            [
                A.augmentations.crops.transforms.RandomResizedCrop(
                    *self.img_size, p=0.5
                ),
                A.augmentations.MotionBlur(p=0.1),
                A.augmentations.transforms.ColorJitter(),
                A.augmentations.geometric.rotate.SafeRotate(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.augmentations.geometric.rotate.RandomRotate90(p=0.5),
            ]
        )

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.x)

    def __getitem__(self, index):
        img = cv2.cvtColor(cv2.imread(self.x[index]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.y[index], cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_CUBIC)

        if self.is_train:
            if self.is_train:
                transformed = self.transform(image=img, mask=mask)
                img = transformed["image"]
                mask = transformed["mask"]

        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        img = self.img_totensor(img.copy())
        mask = self.mask_totensor(mask.copy()).unsqueeze(0)

        if self.is_train and self.is_boundary:
            return img, mask, _bextraction(img)
        return img, mask
