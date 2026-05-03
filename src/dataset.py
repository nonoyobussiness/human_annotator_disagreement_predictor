import torch
from torch.utils.data import Dataset
import numpy as np

class CIFAR10HDataset(Dataset):
    def __init__(self, images, soft_labels, indices):
        self.images = images[indices]
        self.soft_labels = soft_labels[indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        img = img.reshape(3, 32, 32)
        img = torch.tensor(img, dtype=torch.float32) / 255.0

        label = torch.tensor(self.soft_labels[idx], dtype=torch.float32)

        return img, label