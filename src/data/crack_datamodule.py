import os
from typing import Optional

from torch.utils.data import Dataset

from .components import CustomDataset
from .components.crack_dataset import CrackDataset
from .data_module import BaseDataModule

__all__ = ["CrackDataModule"]


class CrackDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        dataset: Dataset,
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

    def setup(self, stage: Optional[str] = None):
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

        if self.trainer:
            if stage == "fit" or stage is None:
                self.data_train = self.hparams.dataset(
                    *path_list("train"), is_train=True
                )
                self.data_val = self.hparams.dataset(*path_list("validation"))
            if stage == "test" or stage is None:
                self.data_test = self.hparams.dataset(*path_list("test"))
