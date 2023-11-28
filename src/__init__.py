import os
import numpy as np
import torch
import seaborn as sns

sns.set()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
GPU_ID = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

SEED = 1234

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else None
