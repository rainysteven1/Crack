import time
import numpy as np
import torch
import torchmetrics

from logging import Logger
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from src import DEVICE
from .dataset import CrackDataset
from .utils import build_model, log_model_summary

input_dim = 3


def predict(
    logger: Logger,
    category: str,
    load_model_dir: str,
    path_dict: dict,
    data_attributes: dict,
    test_settings: dict,
):
    batch_size = test_settings["batch_size"]
    mode = test_settings["metrics_mode"]
    model = build_model(category)
    model.load_state_dict(torch.load(load_model_dir, map_location=DEVICE))
    log_model_summary(logger, model, batch_size, device=DEVICE, **data_attributes)
    logger.info(f"Loading Model State from {load_model_dir}")

    dataset = CrackDataset(
        path_dict["test"]["image"],
        path_dict["test"]["mask"],
        **data_attributes,
        is_augment=False,
    )
    loader = DataLoader(dataset, batch_size, num_workers=1, pin_memory=1)
    x_all = torch.empty(
        [0, 3, *data_attributes.values()],
        device=DEVICE,
        dtype=torch.float32,
    )
    y_true_all = torch.empty(
        [0, 1, *data_attributes.values()],
        device=DEVICE,
        dtype=torch.float32,
    )
    y_pred_all = torch.empty_like(
        y_true_all,
        device=DEVICE,
        dtype=torch.float32,
    )

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, dtype=torch.float32)
            y = y.to(DEVICE, dtype=torch.float32)
            outputs = model(x)
            y_pred = y_pred if not isinstance(outputs, list) else outputs[-1]
            x_all = torch.cat((x_all, x), dim=0)
            y_true_all = torch.cat((y_true_all, y), dim=0)
            y_pred_all = torch.cat((y_pred_all, y_pred), dim=0)

    def log_process(function):
        logger.info("Metric(Prediction):")
        log_test_metrics(logger, *function(y_true_all, y_pred_all))

    if mode == "gpu":
        log_process(calculate_test_metrics_gpu)
    x_all = x_all.detach().cpu().numpy()
    y_true_all = y_true_all.detach().cpu().numpy()
    y_pred_all = y_pred_all.detach().cpu().numpy()
    if mode == "cpu":
        log_process(calculate_test_metrics_cpu)

    duration = time.time() - start_time
    logger.info("Duration - %ds" % (duration))
    logger.info("Predict Done!")
    return {
        "Image": {"data": x_all, "is_gray": False},
        "Mask": {"data": y_true_all, "is_gray": True},
        "Prediction": {"data": y_pred_all, "is_gray": True},
    }


def calculate_test_metrics_cpu(y_true: np.ndarray, y_pred: np.ndarray):
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

    return (
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


def calculate_test_metrics_gpu(y_true: torch.Tensor, y_pred: torch.Tensor):
    yy_true = y_true.view(-1)
    yy_pred = y_pred.view(-1)

    accuracy = torchmetrics.functional.accuracy(yy_pred, yy_true, task="binary")

    precision = torchmetrics.functional.precision(yy_pred, yy_true, task="binary")
    recall = torchmetrics.functional.recall(yy_pred, yy_true, task="binary")
    f1_score = torchmetrics.functional.f1_score(yy_pred, yy_true, task="binary")
    sensitivity = recall
    specificity = torchmetrics.functional.specificity(yy_pred, yy_true, task="binary")

    AUC = torchmetrics.functional.auroc(yy_pred, yy_true.int(), task="binary")
    IOU = (precision * recall) / (precision + recall - precision * recall)

    return (
        accuracy,
        precision,
        recall,
        f1_score,
        sensitivity,
        specificity,
        AUC,
        IOU,
        None,
    )


def log_test_metrics(
    logger: Logger,
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
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1_score:.4f}")
    logger.info(f"Sensitivity: {sensitivity:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")
    logger.info(f"AUC: {AUC:.4f}")
    logger.info(f"IOU: {IOU:.4f}")
    if report:
        logger.info(f"Report:\n{report}")
