import os, time
import numpy as np
import pandas as pd
import torch

from logging import Logger
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import BinaryAccuracy

from src import DEVICE
from core.losses import *
from core.metrics import IOU
from core import MODEL_DICT
from dataset import CrackDataset
from utils import log_model_summary, MetricTracker


class MetricRecord:
    def __init__(self) -> None:
        self.loss = MetricTracker()
        self.IOU = MetricTracker()
        self.acc = MetricTracker()

        self.reset()

    def reset(self):
        self.loss.reset()
        self.IOU.reset()
        self.acc.reset()

    def update(self, loss, IOU, acc, n=1):
        self.loss.update(loss, n)
        self.IOU.update(IOU, n)
        self.acc.update(acc, n)

    def get_metrics(self):
        return [self.loss.get_avg(), self.acc.get_avg(), self.IOU.get_avg()]


def lr_schedule(epoch):
    scale_factor = 1
    if epoch > 150:
        scale_factor *= 2 ** (-1)
    elif epoch > 80:
        scale_factor *= 2 ** (-1)
    elif epoch > 50:
        scale_factor *= 2 ** (-1)
    elif epoch > 30:
        scale_factor *= 2 ** (-1)
    return scale_factor


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
    train_loader = DataLoader(train_dataset, batch_size, num_workers=1, pin_memory=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size, num_workers=1, pin_memory=True
    )
    train_length = len(train_loader)
    validation_length = len(validation_loader)
    logger.info("number of train data batch: %d" % train_length)
    logger.info("number of validation data batch: %d" % validation_length)
    train_metric_record = MetricRecord()
    validation_metric_record = MetricRecord()

    model = MODEL_DICT.get(category)(input_dim=3, output_dim=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0035, eps=1e-7, amsgrad=True)
    scheduler = LambdaLR(optimizer, lr_schedule, verbose=True)
    criterion = DiceLoss()
    metric = BinaryAccuracy().to(DEVICE)
    log_model_summary(
        logger, model, batch_size, input_dim=3, device=DEVICE, **data_attributes
    )

    best_loss = float("inf")
    msg_list = list()

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        model.train()
        train_metric_record.reset()
        for x, y in train_loader:
            x = x.to(DEVICE, dtype=torch.float32)
            y = y.view(-1).to(DEVICE, dtype=torch.float32)
            optimizer.zero_grad()
            y_pred = model(x).view(-1)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_metric_record.update(
                loss.item(), IOU(y_pred, y).item(), metric(y_pred, y).item()
            )

        model.eval()
        validation_metric_record.reset()
        with torch.no_grad():
            for x, y in validation_loader:
                x = x.to(DEVICE, dtype=torch.float32)
                y = y.view(-1).to(DEVICE, dtype=torch.float32)
                y_pred = model(x).view(-1)
                loss = criterion(y_pred, y)
                validation_metric_record.update(
                    loss.item(),
                    IOU(y_pred, y).item(),
                    metric(y_pred, y).item(),
                )

        duration = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]
        logger_msg = (
            "Epoch: %5d/%d - %ds - lr: %.4e - loss: %.4f - acc: %.4f - IOU: %.4f - val_loss: %.4f - val_acc: %.4f - val_IOU: %.4f"
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
        if validation_loss < best_loss:
            checkpoint_path = os.path.join(load_model_dir, "checkpoint.pth")
            logger.info(
                f"Valid loss improved from {best_loss:2.4f} to {validation_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            )
            best_loss = validation_loss
            torch.save(model.state_dict(), checkpoint_path)

        scheduler.step()

    df = pd.DataFrame(msg_list)
    df.to_csv(loss_csv, encoding="utf8", index=False)
    torch.save(model.state_dict, os.path.join(load_model_dir, "model.pth"))
    logger.info("Train Done!")
