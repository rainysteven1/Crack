import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_data(images, masks, size, category, is_original=True):
    title = "{} {{}} ({})".format("Original" if is_original else "Augmented", category)

    def plot_types(source, height, is_images=True):
        source = np.transpose(source, (0, 2, 3, 1))  # NHWC
        fig, axes = plt.subplots(1, size, figsize=(13, height))
        fig.suptitle(title.format("Images" if is_images else "Masks"), fontsize=15)
        axes = axes.flatten()
        for img, ax in zip(source[:size], axes[:size]):
            if is_images:
                ax.imshow(img)
            else:
                ax.imshow(np.squeeze(img, -1), cmap="gray")
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    plot_types(images, 2.5)
    plot_types(masks, 3.0, False)


def plot_loss(loss_csv, figure_path):
    df = pd.read_csv(loss_csv)
    train_loss = None if "loss" not in df.columns else df["loss"]
    validation_loss = df["val_loss"]

    _, axes = plt.subplots(1, 1, figsize=(13, 4))
    if train_loss:
        axes.plot(train_loss, label="training")
    axes.plot(validation_loss, label="validation")
    axes.set_title("Loss Curve")
    axes.set_xlabel("epochs")
    axes.set_ylabel("loss")
    axes.legend()

    plt.savefig(figure_path)


def plot_metrics(loss_csv, figure_path):
    df = pd.read_csv(loss_csv)
    train_acc = df["acc"]
    validation_acc = df["val_acc"]
    train_IOU = df["IOU"]
    validation_IOU = df["val_IOU"]

    _, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes = axes.flatten()

    axes[0].plot(train_acc, label="training")
    axes[0].plot(validation_acc, label="validation")
    axes[0].set_title("Accuracy Curve")
    axes[0].set_xlabel("epochs")
    axes[0].set_ylabel("accuracy")
    axes[0].legend()

    axes[1].plot(train_IOU, label="training")
    axes[1].plot(validation_IOU, label="validation")
    axes[1].set_title("IOU Curve")
    axes[1].set_xlabel("epochs")
    axes[1].set_ylabel("IOU")
    axes[1].legend()

    plt.savefig(figure_path)


def visualize(data_dict, indices, figure_path):
    _, axes = plt.subplots(1, 3, figsize=(13, 4))
    for index in indices:
        for idx, (key, value) in enumerate(data_dict.items()):
            axes[idx].imshow(
                np.transpose(value["data"][index], (1, 2, 0)),
                cmap="gray" if value["is_gray"] else "viridis",
            )
            axes[idx].axis("off")
            axes[idx].set_title(f"[{key}]", loc="center")
        plt.tight_layout()
        plt.savefig(os.path.join(figure_path, f"result_{index}.png"))


def visualize_dict(data_dict, indices, figure_path):
    _, axes = plt.subplots(1, 4, figsize=(10, 4))
    for index in indices:
        for idx, (key, value) in enumerate(data_dict.items()):
            if "data" in value:
                axes[idx].imshow(
                    np.transpose(value["data"][index], (1, 2, 0)),
                    cmap="gray" if value["is_gray"] else "viridis",
                )
                axes[idx].axis("off")
                axes[idx].set_title(f"[{key}]", loc="center")
            else:
                for item in value.items():
                    axes[idx].imshow(
                        np.transpose(item[1]["data"][index], (1, 2, 0)),
                        cmap="gray" if item[1]["is_gray"] else "viridis",
                    )
                    axes[idx].axis("off")
                    axes[idx].set_title(f"[{item[0]}]", loc="center")
                    idx += 1
        plt.tight_layout()
        plt.savefig(os.path.join(figure_path, f"result_{index}.png"))
