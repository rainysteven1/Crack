import io, os, sys, time, logging
import numpy as np
import pandas as pd
import torch
import torchmetrics

from typing import Literal
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchsummary import summary
from torchmetrics.classification import BinaryAccuracy
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from src import DEVICE, GPU_NAME
from dataset import CrackDataset
from core.losses import DiceLoss
from core.metrics import IOU, MetricTracker
from core.unet import *
from core.resunet import *
from core.resunet_pp import *


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


class Process:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        batch_height: int,
        batch_width: int,
        logger: logging.Logger,
        load_model_dir: str = "",
    ) -> None:
        self.device = DEVICE
        self.gpu_name = GPU_NAME
        self.logger = logger
        self.load_model_dir = load_model_dir

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_height = batch_height
        self.batch_width = batch_width

        self.model = ResUNetPlusPlus(self.input_dim, self.output_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0035, eps=1e-7, amsgrad=True
        )
        self.scheduler = LambdaLR(self.optimizer, lr_schedule, verbose=True)
        self.criterion = DiceLoss()
        self.metric = BinaryAccuracy().to(DEVICE)

    def train(
        self,
        path_dict: dict,
        batch_size: int,
        epochs: int,
        train_split: int,
        loss_csv: str,
    ):
        self.log_model_summary(batch_size)
        N_train = 1896
        train_indices = np.random.choice(N_train, train_split, replace=False)
        validation_indices = np.delete(np.arange(N_train), train_indices)
        train_dataset = CrackDataset(
            [path_dict["train"]["image"][i] for i in train_indices],
            [path_dict["train"]["mask"][i] for i in train_indices],
            self.batch_height,
            self.batch_width,
            is_augment=True,
        )
        validation_dataset = CrackDataset(
            [path_dict["train"]["image"][i] for i in validation_indices],
            [path_dict["train"]["mask"][i] for i in validation_indices],
            self.batch_height,
            self.batch_width,
            is_augment=False,
        )
        train_loader = DataLoader(
            train_dataset, batch_size, num_workers=1, pin_memory=True
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size, num_workers=1, pin_memory=True
        )
        train_length = len(train_loader)
        validation_length = len(validation_loader)
        self.logger.info("number of train data batch: %d" % train_length)
        self.logger.info("number of validation data batch: %d" % validation_length)

        train_loss = MetricTracker()
        train_IOU = MetricTracker()
        train_acc = MetricTracker()
        validation_loss = MetricTracker()
        validation_IOU = MetricTracker()
        validation_acc = MetricTracker()

        best_loss = float("inf")
        msg_list = list()

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            self.model.train()
            train_loss.reset()
            train_IOU.reset()
            train_acc.reset()
            for x, y in train_loader:
                x = x.to(self.device, dtype=torch.float32)
                y = y.view(-1).to(self.device, dtype=torch.float32)
                self.optimizer.zero_grad()
                y_pred = self.model(x).view(-1)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                train_loss.update(loss.item())
                train_IOU.update(IOU(y_pred, y).item())
                train_acc.update(self.metric(y_pred, y).item())

            self.model.eval()
            validation_loss.reset()
            validation_IOU.reset()
            validation_acc.reset()
            with torch.no_grad():
                for x, y in validation_loader:
                    x = x.to(self.device, dtype=torch.float32)
                    y = y.view(-1).to(self.device, dtype=torch.float32)
                    y_pred = self.model(x).view(-1)
                    loss = self.criterion(y_pred, y)
                    validation_loss.update(loss.item())
                    validation_IOU.update(IOU(y_pred, y).item())
                    validation_acc.update(self.metric(y_pred, y).item())

            duration = time.time() - start_time
            lr = self.optimizer.param_groups[0]["lr"]
            logger_msg = (
                "Epoch: %5d/%d - %ds - lr: %.4e - loss: %.4f - acc: %.4f - IOU: %.4f - val_loss: %.4f - val_acc: %.4f - val_IOU: %.4f"
                % (
                    epoch,
                    epochs,
                    duration,
                    lr,
                    train_loss.get_avg(),
                    train_acc.get_avg(),
                    train_IOU.get_avg(),
                    validation_loss.get_avg(),
                    validation_acc.get_avg(),
                    validation_IOU.get_avg(),
                )
            )
            self.logger.info(logger_msg)
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

            if validation_loss.get_avg() < best_loss:
                checkpoint_path = os.path.join(self.load_model_dir, "checkpoint.pth")
                self.logger.info(
                    f"Valid loss improved from {best_loss:2.4f} to {validation_loss.get_avg():2.4f}. Saving checkpoint: {checkpoint_path}"
                )
                best_loss = validation_loss.get_avg()
                torch.save(self.model.state_dict(), checkpoint_path)

            self.scheduler.step()

        df = pd.DataFrame(msg_list)
        df.to_csv(loss_csv, encoding="utf8", index=False)
        torch.save(
            self.model.state_dict, os.path.join(self.load_model_dir, "model.pth")
        )
        self.logger.info("Train Done!")

    def predict(
        self,
        path_dict: dict,
        batch_size: int,
        calulate_metrics_mode: Literal["cpu", "gpu"] = "gpu",
    ):
        self.model.load_state_dict(
            torch.load(self.load_model_dir, map_location=self.device)
        )
        self.log_model_summary(batch_size)
        self.logger.info(f"Loading Model State from {self.load_model_dir}")

        dataset = CrackDataset(
            path_dict["test"]["image"],
            path_dict["test"]["mask"],
            self.batch_height,
            self.batch_width,
            is_augment=False,
        )
        loader = DataLoader(dataset, batch_size, num_workers=1, pin_memory=1)
        x_all = torch.empty(
            (
                0,
                3,
                self.batch_height,
                self.batch_width,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        y_true_all = torch.empty(
            (
                0,
                self.output_dim,
                self.batch_height,
                self.batch_width,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        y_pred_all = torch.empty_like(
            y_true_all,
            device=self.device,
            dtype=torch.float32,
        )

        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32)
                y_pred = self.model(x)
                x_all = torch.cat((x_all, x), dim=0)
                y_true_all = torch.cat((y_true_all, y), dim=0)
                y_pred_all = torch.cat((y_pred_all, y_pred), dim=0)

        if calulate_metrics_mode == "gpu":
            self.calculate_test_metrics_gpu(y_true_all, y_pred_all)
        x_all = x_all.detach().cpu().numpy()
        y_true_all = y_true_all.detach().cpu().numpy()
        y_pred_all = y_pred_all.detach().cpu().numpy()
        if calulate_metrics_mode == "cpu":
            self.calculate_test_metrics_cpu(y_true_all, y_pred_all)

        duration = time.time() - start_time
        self.logger.info("Duration - %ds" % (duration))
        self.logger.info("Predict Done!")
        return {
            "Image": {"data": x_all, "is_gray": False},
            "Mask": {"data": y_true_all, "is_gray": True},
            "Prediction": {"data": y_pred_all, "is_gray": True},
        }

    def calculate_test_metrics_cpu(self, y_true: np.ndarray, y_pred: np.ndarray):
        thresholds = 0.5
        yy_true = (y_true > thresholds).flatten()
        yy_pred = (y_pred > thresholds).flatten()

        report = classification_report(yy_true, yy_pred, output_dict=True)
        accuracy = accuracy_score(yy_true, yy_pred)

        precision = report["True"]["precision"]
        recall = report["True"]["recall"]
        f1_score = report["True"]["f1-score"]
        sensitivity = recall
        specificity = report["False"]["recall"]

        AUC = roc_auc_score(y_true.flatten(), y_pred.flatten())
        IOU = (precision * recall) / (precision + recall - precision * recall)

        self.log_test_metrics(
            accuracy,
            precision,
            recall,
            f1_score,
            sensitivity,
            specificity,
            AUC,
            IOU,
            report,
        )

    def calculate_test_metrics_gpu(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        yy_true = torch.flatten(y_true)
        yy_pred = torch.flatten(y_pred)

        accuracy = torchmetrics.functional.accuracy(yy_pred, yy_true, task="binary")

        precision = torchmetrics.functional.precision(yy_pred, yy_true, task="binary")
        recall = torchmetrics.functional.recall(yy_pred, yy_true, task="binary")
        f1_score = torchmetrics.functional.f1_score(yy_pred, yy_true, task="binary")
        sensitivity = recall
        specificity = torchmetrics.functional.specificity(
            yy_pred, yy_true, task="binary"
        )

        AUC = torchmetrics.functional.auroc(yy_pred, yy_true.int(), task="binary")
        IOU = (precision * recall) / (precision + recall - precision * recall)

        self.log_test_metrics(
            accuracy.item(),
            precision.item(),
            recall.item(),
            f1_score.item(),
            sensitivity.item(),
            specificity.item(),
            AUC.item(),
            IOU.item(),
        )

    def log_model_summary(self, batch_size: int):
        output = io.StringIO()
        sys.stdout = output
        summary(
            self.model,
            (self.input_dim, self.batch_height, self.batch_width),
            batch_size,
            self.device,
        )
        sys.stdout = sys.__stdout__
        summary_output = output.getvalue()
        self.logger.info("Model:\n{}".format(summary_output))

    def log_test_metrics(
        self,
        accuracy,
        precision,
        recall,
        f1_score,
        sensitivity,
        specificity,
        AUC,
        IOU,
        report=None,
    ):
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1-Score: {f1_score:.4f}")
        self.logger.info(f"Sensitivity: {sensitivity:.4f}")
        self.logger.info(f"Specificity: {specificity:.4f}")
        self.logger.info(f"AUC: {AUC:.4f}")
        self.logger.info(f"IOU: {IOU:.4f}")
        if report:
            self.logger.info(f"Report:\n{report}")
