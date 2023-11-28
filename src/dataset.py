import cv2
import numpy as np
from torch.utils.data import Dataset
from albumentations import (
    Compose,
    Flip,
    Rotate,
    OneOf,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
)


class CrackDataset(Dataset):
    def __init__(self, x_set, y_set, batch_height, batch_width, is_augment=False):
        super().__init__()
        self.x = x_set
        self.y = y_set
        self.batch_height = batch_height
        self.batch_width = batch_width
        self.is_augment = is_augment
        self.augmentations = Compose(
            [
                Flip(p=0.7),
                Rotate(p=0.7),
                OneOf([RandomContrast(), RandomGamma(), RandomBrightness()], p=0.3),
                OneOf(
                    [
                        ElasticTransform(
                            alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                        ),
                        GridDistortion(),
                        OpticalDistortion(distort_limit=2, shift_limit=0.5),
                    ],
                    p=0.3,
                ),
            ]
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        batch_x = self.x[index]
        batch_y = self.y[index]

        batch_x = np.array(
            [
                cv2.resize(
                    cv2.cvtColor(cv2.imread(batch_x, -1), cv2.COLOR_BGR2RGB),
                    dsize=(self.batch_width, self.batch_height),
                )
            ]
        )
        batch_y = np.array(
            [
                (
                    cv2.resize(
                        cv2.imread(batch_y, -1),
                        dsize=(self.batch_width, self.batch_height),
                    )
                    > 0
                ).astype(np.uint8)
            ]
        )

        if self.is_augment:
            augs = [
                self.augmentations(image=x, mask=y) for x, y in zip(batch_x, batch_y)
            ]
            batch_x = np.array([aug["image"] for aug in augs])
            batch_y = np.array([aug["mask"] for aug in augs])

        batch_x = np.transpose(batch_x[0], (2, 0, 1))
        return batch_x / 255, batch_y / 1
