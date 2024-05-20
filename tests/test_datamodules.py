from pathlib import Path

import pytest
import torch

from src.data.crack_datamodule import CrackDataModule


@pytest.mark.parametrize("batch_size", [4, 8])
def test_crack_datamodule(batch_size: int) -> None:
    """Tests `CrackDataModule` to verify that it can be downloaded correctly, that the
    necessary attributes were created (e.g., the dataloader objects), and that dtypes
    and batch sizes correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"
    category_list = ["train", "validation", "test"]

    dm = CrackDataModule(
        data_dir=data_dir,
        img_size=256,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "crack500").exists()
    for category in category_list:
        assert Path(data_dir, f"crack500/{category}").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
