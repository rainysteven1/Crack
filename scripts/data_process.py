import argparse
import asyncio
import io
import os
import re
from datetime import datetime
from typing import List, Optional, Tuple

import aiofiles
import matplotlib.pyplot as plt
import polars as pl
from polars.expr.expr import Expr

_ROOT_DIR = "logs/train/runs"
_TAGS_LOG = "tags.log"
_TIMEFORMAT = "%Y-%m-%d_%H-%M-%S"

_CONFIGS = {
    "PR_Curve": {
        "filter_condition": pl.col("Recall").gt(0.01),
        "title": "Precision-Recall Curve",
        "x_label": "Recall",
        "y_label": "Precision",
    },
    "ROC_Curve": {
        "title": "Receiver Operating Characteristic (ROC) Curve",
        "x_label": "fpr",
        "y_label": "tpr",
    },
}


def _get_work_dir() -> str:
    sorted_dirs = sorted(
        os.listdir(_ROOT_DIR), key=lambda x: datetime.strptime(x, _TIMEFORMAT)
    )
    return sorted_dirs[0]


async def _read_tags_log(sub_dir: str) -> Optional[Tuple[str, str]]:
    pattern = r"'(.*?)'"
    file_path = os.path.join(sub_dir, _TAGS_LOG)
    if os.path.isfile(file_path):
        async with aiofiles.open(file_path, "r") as file:
            line = await file.readline()
        return re.findall(pattern, line)[-1], sub_dir
    return None


async def _read_csv(sub_dir: str, csv_name: str) -> pl.DataFrame:
    file_path = os.path.join(sub_dir, "figure/data", csv_name)
    if os.path.isfile(file_path):
        async with aiofiles.open(file_path, "rb") as file:
            content = await file.read()
        return pl.read_csv(content)
    return None


async def _get_csv_datas(
    model_names_tuple: Tuple[str, str], csv_name: str, filter_condition: Optional[Expr]
) -> List[pl.DataFrame]:
    dfs = list()

    tasks = [
        (model_name, _read_csv(sub_dir, csv_name))
        for model_name, sub_dir in model_names_tuple
    ]

    for model_name, task in tasks:
        df = await task
        if filter_condition is not None:
            df = df.filter(filter_condition)
        new_df = df.with_columns(pl.lit(model_name).alias("Classifier"))
        dfs.append(new_df)
    return dfs


def _plot_curves(
    dfs: List[pl.DataFrame],
    model_names_tuple: Tuple[str, str],
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    for df, (model_name, _) in zip(dfs, model_names_tuple):
        ax.plot(df[x_label], df[y_label], linewidth=1, label=model_name)

    ax.grid(True)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=3,
            fancybox=True,
            shadow=True,
        )
    ax.legend(title="Classifier", fontsize=12, loc="best")

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    return fig


async def main(work_dir: str) -> None:
    sub_dirs = [
        os.path.join(work_dir, sub_dir)
        for sub_dir in os.listdir(work_dir)
        if os.path.isdir(os.path.join(work_dir, sub_dir))
    ]
    tasks = [_read_tags_log(sub_dir) for sub_dir in sub_dirs]
    model_names_tuple = await asyncio.gather(*tasks)
    for key, value in _CONFIGS.items():
        fig = _plot_curves(
            await _get_csv_datas(
                model_names_tuple,
                f"{key}.csv",
                value.pop("filter_condition", None),
            ),
            model_names_tuple,
            **value,
        )
        fig.savefig(os.path.join(work_dir, f"{key}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default=_get_work_dir())
    args = parser.parse_args()
    work_dir = os.path.join(os.getcwd(), _ROOT_DIR, args.work_dir)
    asyncio.run(main(work_dir))
