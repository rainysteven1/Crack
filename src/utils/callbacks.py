import os
from typing import Dict

import pandas as pd
from lightning import Callback, LightningModule, Trainer
from torchmetrics import Metric, MetricCollection

__all__ = ["CurveCallback"]


def _pr_curve_data(metric: Metric) -> Dict:
    curve_computed = metric.compute()
    return {
        "Recall": curve_computed[1].detach().cpu(),
        "Precision": curve_computed[0].detach().cpu(),
    }


def _roc_curve_data(metric: Metric) -> Dict:
    curve_computed = metric.compute()
    return {
        "fpr": curve_computed[0].detach().cpu(),
        "tpr": curve_computed[1].detach().cpu(),
    }


_CURVE_DICT = {
    "PR_Curve": {"title": "Precision-Recall Curve", "fn": _pr_curve_data},
    "ROC_Curve": {
        "title": "Receiver Operating Characteristic (ROC) Curve",
        "fn": _roc_curve_data,
    },
}


class CurveCallback(Callback):

    def __init__(self, data_path: str, figure_path: str):
        self.data_path = data_path
        self.figure_path = figure_path

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        assert pl_module.test_curves and isinstance(
            pl_module.test_curves, MetricCollection
        )

        for name, value in pl_module.test_curves.items():
            if value.compute():
                curve_dict = _CURVE_DICT.get(name)

                df = pd.DataFrame(curve_dict["fn"](value))
                df.to_csv(os.path.join(self.data_path, f"{name}.csv"), index=False)

                fig, ax = value.plot(score=True)
                ax.set_title(curve_dict.get("title"))
                fig.savefig(os.path.join(self.figure_path, f"{name}.png"))

        pl_module.test_curves.reset()
