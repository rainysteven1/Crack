from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchmetrics.classification as C
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MetricCollection

__all__ = ["BaseModule"]


class BaseModule(LightningModule):
    """A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        loss_weights: Optional[List[float]],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `LitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["net", "criterion"], logger=False)

        self.net = net

        # loss function
        self.criterion = criterion

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_metrics = MetricCollection(
            {
                "acc": C.BinaryAccuracy(),
                "IoU": C.BinaryJaccardIndex(),
            },
            prefix="train/",
        )

        self.val_metrics = MetricCollection(
            {
                "acc": C.BinaryAccuracy(),
                "IoU": C.BinaryJaccardIndex(),
            },
            prefix="val/",
        )

        self.test_metrics = MetricCollection(
            {
                "Accuracy": C.BinaryAccuracy(),
                "Precision": C.BinaryPrecision(),
                "Recall": C.BinaryRecall(),
                "F1-Score": C.BinaryF1Score(),
                "Specificity": C.BinarySpecificity(),
                "AUROC": C.BinaryAUROC(),
                "IoU": C.BinaryJaccardIndex(),
            },
            prefix="test/",
        )

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_metrics.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        length = len(batch)
        logits = self.forward(batch[0])
        if not isinstance(logits, (list, tuple)):
            pred = F.sigmoid(logits)
            loss = self.criterion(pred, batch[1])
        else:
            preds = list(map(F.sigmoid, logits))
            pred = preds[-1]
            if self.is_train:
                if length == 2:
                    losses = [self.criterion(pred, batch[1]) for pred in preds]
                else:
                    assert len(preds) == length - 1
                    losses = [
                        self.criterion(pred, target)
                        for pred, target in zip(reversed(preds), batch[1:])
                    ]

                assert self.hparams.loss_weights and len(losses) == len(
                    self.hparams.loss_weights
                )
                loss = sum(
                    [
                        loss * weight
                        for loss, weight in zip(losses, self.hparams.loss_weights)
                    ]
                )
            else:
                loss = self.criterion(pred, batch[1])
        return loss, pred, batch[1].int()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images
            and target labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, pred, target = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_metrics(pred, target)

        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        for name, metric in self.train_metrics.items():
            self.log(name, metric, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def on_validation_start(self) -> None:
        self.is_train = False

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images
            and target labels.
        :param batch_idx: The index of the current batch.
        """
        loss, pred, target = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_metrics(pred, target)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, metric in self.val_metrics.items():
            self.log(name, metric, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.is_train = False
        acc = self.val_metrics.compute().get("val/acc")  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images
            and target labels.
        :param batch_idx: The index of the current batch.
        """
        loss, pred, target = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_metrics(pred, target)

        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        for name, metric in self.test_metrics.items():
            self.log(name, metric, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate),
        validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust
        something about them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally you'd need one. But in the case of GANs or similar you
        might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        if getattr(self.trainer.model, "get_params", None):
            keywords = self.hparams.optimizer.keywords
            params = self.trainer.model.get_params(keywords)
            del keywords["lr"]
            del keywords["weight_decay"]
        else:
            params = self.trainer.model.parameters()

        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = BaseModule(None, None, None, None)
