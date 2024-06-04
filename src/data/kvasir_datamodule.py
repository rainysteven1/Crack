import os

from .components import CustomDataset
from .data_module import BaseDataModule


class KvasirDataModule(BaseDataModule):
    def __init__(
        self,
        train_data_dir: str,
        test_data_dir: str,
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

    def setup(self, stage=None):
        def path_list(category, data_dir):
            set_list = list()
            dir_list = ["image", "groundtruth"]
            for dir_name in dir_list:
                dir = os.path.join(data_dir, category, dir_name)
                set_list.append(
                    [
                        os.path.join(dir, file_name)
                        for file_name in sorted(os.listdir(dir))
                    ]
                )
            return set_list

        if self.trainer:
            self.data_train = CustomDataset(
                *path_list("train", self.hparams.train_data_dir), transform=None
            )
            self.data_val = CustomDataset(
                *path_list("validation", self.hparams.train_data_dir), transform=None
            )
            self.data_test = CustomDataset(
                *path_list("test", self.hparams.test_data_dir), transform=None
            )
