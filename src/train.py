import os, time
import numpy as np
import pandas as pd
import torch

from logging import Logger
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex

from src import DEVICE
from .dataset import CrackDataset
from .utils import *


class MetricRecord:
    def __init__(self) -> None:
        self.loss = MetricTracker()
        self.acc = MetricTracker()
        self.IoU = MetricTracker()
        self.reset()

    def reset(self):
        self.loss.reset()
        self.acc.reset()
        self.IoU.reset()

    def update(self, loss, IoU, acc, n=1):
        self.loss.update(loss, n)
        self.acc.update(acc, n)
        self.IoU.update(IoU, n)

    def get_metrics(self):
        return [self.loss.get_avg(), self.acc.get_avg(), self.IoU.get_avg()]


def train(
    logger: Logger,
    category: str,
    load_model_dir: str,
    path_dict: dict,
    loss_csv: str,
    data_attributes: dict,
    train_settings: dict,
):
    batch_size = train_settings["batch_size"]
    epochs = train_settings["N_epochs"]

    N_train = 1896
    train_indices = np.random.choice(
        N_train, train_settings["train_split"], replace=False
    )
    validation_indices = np.delete(np.arange(N_train), train_indices)
    train_dataset = CrackDataset(
        [path_dict["train"]["image"][i] for i in train_indices],
        [path_dict["train"]["mask"][i] for i in train_indices],
        **data_attributes,
        is_augment=True,
    )
    validation_dataset = CrackDataset(
        [path_dict["train"]["image"][i] for i in validation_indices],
        [path_dict["train"]["mask"][i] for i in validation_indices],
        **data_attributes,
        is_augment=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    validation_loader = DataLoader(
        validation_dataset, batch_size, shuffle=False, num_workers=1, pin_memory=True
    )
    train_length = len(train_loader)
    validation_length = len(validation_loader)
    logger.info("number of train data batch: %d" % train_length)
    logger.info("number of validation data batch: %d" % validation_length)
    train_metric_record = MetricRecord()
    validation_metric_record = MetricRecord()

    model = build_model(category)
    log_model_summary(logger, model, batch_size, device=DEVICE, **data_attributes)
    optimizer = get_optimizer(train_settings["optimizer"], model)
    criterion = get_criterion(train_settings["criterion"])
    IoU = BinaryJaccardIndex().to(DEVICE)
    acc = BinaryAccuracy().to(DEVICE)

    best_loss = float("inf")
    early_stopping_count = 0
    msg_list = list()
    scheduler = get_scheduler(train_settings["scheduler"], optimizer, epochs)
    early_stopping_patience = train_settings["early_stopping_patience"]

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        model.train()
        train_metric_record.reset()
        for x, y in train_loader:
            x = x.to(DEVICE, dtype=torch.float32)
            y = y.to(DEVICE, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(x)
            if not isinstance(outputs, list):
                y_pred = outputs
                loss = criterion(y_pred, y)
            else:
                loss = 0
                for output in outputs:
                    loss += criterion(output, y)
                loss /= len(outputs)
                y_pred = outputs[-1]
            loss.backward()
            optimizer.step()
            train_metric_record.update(
                loss.item(), IoU(y_pred, y).item(), acc(y_pred, y).item()
            )

        model.eval()
        validation_metric_record.reset()
        with torch.no_grad():
            for x, y in validation_loader:
                x = x.to(DEVICE, dtype=torch.float32)
                y = y.to(DEVICE, dtype=torch.float32)
                y_pred = model(x)
                if not isinstance(y_pred, list):
                    y_pred = y_pred
                    loss = criterion(y_pred, y)
                else:
                    loss = 0
                    for output in y_pred:
                        loss += criterion(output, y)
                    loss /= len(outputs)
                    y_pred = y_pred[-1]
                validation_metric_record.update(
                    loss.item(),
                    IoU(y_pred, y).item(),
                    acc(y_pred, y).item(),
                )

        duration = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]
        logger_msg = (
            "Epoch: %5d/%d - %ds - lr: %.4e - loss: %.4f - acc: %.4f - IoU: %.4f - val_loss: %.4f - val_acc: %.4f - val_IoU: %.4f"
            % (
                epoch,
                epochs,
                duration,
                lr,
                *train_metric_record.get_metrics(),
                *validation_metric_record.get_metrics(),
            )
        )
        logger.info(logger_msg)
        msg_list.append(
            dict(
                [
                    string.replace(" ", "").split(":")
                    for string in filter(
                        lambda x: x.find(":") != -1, logger_msg.split(" - ")
                    )
                ]
            )
        )

        validation_loss = validation_metric_record.get_metrics()[0]

        def save_best_checkpoint(original_loss):
            checkpoint_path = os.path.join(load_model_dir, "checkpoint.pth")
            logger.info(
                f"Valid loss improved from {original_loss:2.4f} to {validation_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            )
            torch.save(model.state_dict(), checkpoint_path)
            return validation_loss

        if scheduler:
            if not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
            else:
                scheduler.step(validation_loss)

        early_stopping_count += 1
        if validation_loss < best_loss:
            best_loss = save_best_checkpoint(best_loss)
            early_stopping_count = 0
        if (
            early_stopping_patience >= 0
            and early_stopping_count == early_stopping_patience
        ):
            logger.info(
                f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously"
            )
            break

    df = pd.DataFrame(msg_list)
    df.to_csv(loss_csv, encoding="utf8", index=False)
    torch.save(model.state_dict, os.path.join(load_model_dir, "model.pth"))
    logger.info("Train Done!")
