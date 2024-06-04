import cv2
import numpy as np
from torch.utils.data import Dataset

__all__ = ["CustomDataset"]


class CustomDataset(Dataset):
    def __init__(self, x_set: list, y_set: list, transform=None):
        super().__init__()
        self.x = x_set
        self.y = y_set
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        batch_x = cv2.cvtColor(
            cv2.imread(self.x[index], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )
        batch_y = cv2.imread(self.y[index], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augs = self.transform(image=batch_x, mask=batch_y)
            batch_x = augs["image"]
            batch_y = augs["mask"]

        batch_x = (np.transpose(batch_x, (2, 0, 1)) / 255.0).astype(np.float32)
        batch_y = (np.expand_dims(batch_y, axis=0) / 255.0).astype(np.float32)

        return batch_x, batch_y
