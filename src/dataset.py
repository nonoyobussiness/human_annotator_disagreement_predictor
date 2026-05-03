import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

class CIFAR10HDataset(Dataset):
    def __init__(self, images, soft_labels, indices, transform=None):
        self.images = images[indices]
        self.soft_labels = soft_labels[indices]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img.astype(np.uint8))
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        return img, label
