# -*- using: utf-8 -*-
# Author: Yahui Liu <yahui.liu@unitn.it>

import argparse
import asyncio
import glob
import os
import statistics
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from typing import Dict

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="", help="/path/to/segmentation")
args = parser.parse_args()


def _process_image(img_path: str, labels_dict: Dict, labels_dict_lock) -> None:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    labels, counts = np.unique(img, return_counts=True)
    with labels_dict_lock:
        for lab, cnt in zip(labels, counts):
            if lab not in labels_dict:
                labels_dict[lab] = 0
            labels_dict[lab] += cnt


def _calculate_weights(images: str, labels_dict: Dict, labels_dict_lock) -> Dict:
    """
    Reference: https://arxiv.org/abs/1411.4734
    """
    assert os.path.isdir(images)
    img_list = glob.glob(os.path.join(images, "*"))

    with ProcessPoolExecutor() as executor:
        tasks = [
            executor.submit(_process_image, img_path, labels_dict, labels_dict_lock)
            for img_path in img_list
        ]
        for task in tasks:
            task.result()

    total_pixels = sum(labels_dict.values())
    for lab in labels_dict:
        labels_dict[lab] /= float(total_pixels)
    return labels_dict


async def _reverse_weights(w: Dict) -> Dict:
    """Median Frequency Balancing: alpha_c = median_freq/freq(c).
    median_freq is the median of these frequencies
    freq(c) is the number of pixles of class c divided by the total number of pixels in images where c is present
    """
    assert len(w) > 0, "Expected a non-empty weight dict."
    values = list(w.values())
    if len(w) == 1:
        value = 1.0
    elif len(w) == 2:
        value = min(values)
    else:
        value = statistics.median(values)
    for k in w:
        w[k] = value / (w[k] + 1e-10)
    return w


async def main():
    with Manager() as manager:
        labels_dict = manager.dict()
        labels_dict_lock = manager.Lock()
        loop = asyncio.get_event_loop()
        weights = await loop.run_in_executor(
            None, _calculate_weights, args.data_path, labels_dict, labels_dict_lock
        )
        print(weights)
        reversed_weights = await _reverse_weights(weights)
        print(reversed_weights.values())


if __name__ == "__main__":
    asyncio.run(main())
