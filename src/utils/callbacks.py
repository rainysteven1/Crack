import os

from lightning import Callback, LightningModule, Trainer
from torchmetrics import MetricCollection

__all__ = ["CurveCallback"]

_CURVE_DICT = {
    "PR_Curve": "Precision-Recall Curve",
    "ROC_Curve": "Receiver Operating Characteristic (ROC) Curve",
}


class CurveCallback(Callback):

    def __init__(self, figure_dir: str):
        self.figure_dir = figure_dir

        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        assert pl_module.test_curves and isinstance(
            pl_module.test_curves, MetricCollection
        )

        for name, value in pl_module.test_curves.items():
            if value.compute():
                fig, ax = value.plot(score=True)
                ax.set_title(_CURVE_DICT.get(name))
                fig.savefig(os.path.join(self.figure_dir, f"{name}.png"))

        pl_module.test_curves.reset()
