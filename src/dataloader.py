from torch.utils.data import DataLoader
from src.dataset import CIFAR10HDataset
from config import BATCH_SIZE
import numpy as np

def get_dataloaders(images, probs, batch_size=BATCH_SIZE):
    train_idx = np.load('data/processed/train_idx.npy')
    val_idx   = np.load('data/processed/val_idx.npy')
    test_idx  = np.load('data/processed/test_idx.npy')

    train_dataset = CIFAR10HDataset(images, probs, train_idx)
    val_dataset   = CIFAR10HDataset(images, probs, val_idx)
    test_dataset  = CIFAR10HDataset(images, probs, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader