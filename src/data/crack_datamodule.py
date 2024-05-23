import os

import albumentations as A

from .components.dataset import CrackDataset
from .data_module import BaseDataModule


class CrackDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        img_size: int,
        train_batch_size: int,
        test_batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
    ):
        super().__init__(
            train_batch_size,
            test_batch_size,
            num_workers,
            pin_memory,
            persistent_workers,
        )

        self.save_hyperparameters(logger=False)
        # data transformations
        self.train_transforms = A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Flip(p=0.7),
                A.Rotate(p=0.7),
                A.OneOf(
                    [A.RandomBrightnessContrast(), A.RandomGamma()],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                        ),
                        A.GridDistortion(),
                        A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                    ],
                    p=0.3,
                ),
                A.Normalize(),
            ]
        )
        self.val_transforms = A.Compose([A.Resize(img_size, img_size), A.Normalize()])

    def setup(self, stage=None):
        def path_list(category):
            set_list = list()
            dir_list = ["image", "groundtruth"]
            for dir_name in dir_list:
                dir = os.path.join(self.hparams.data_dir, category, dir_name)
                set_list.append(
                    [
                        os.path.join(dir, file_name)
                        for file_name in sorted(os.listdir(dir))
                    ]
                )
            return set_list

        print(self.trainer)

        if self.trainer:
            self.data_train = CrackDataset(
                *path_list("train"), transform=self.train_transforms
            )
            self.data_val = CrackDataset(
                *path_list("validation"), transform=self.val_transforms
            )
            self.data_test = CrackDataset(
                *path_list("test"), transform=self.val_transforms
            )
