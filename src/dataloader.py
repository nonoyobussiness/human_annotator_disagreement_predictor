import numpy as np
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, SEED
from src.dataset import CIFAR10HDataset, eval_transform, train_transform


def _build_generator():
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return generator

def get_dataloaders(images, probs, batch_size=BATCH_SIZE):
    train_idx = np.load('data/processed/train_idx.npy')
    val_idx   = np.load('data/processed/val_idx.npy')
    test_idx  = np.load('data/processed/test_idx.npy')

    train_dataset = CIFAR10HDataset(images, probs, train_idx, transform=train_transform)
    val_dataset   = CIFAR10HDataset(images, probs, val_idx, transform=eval_transform)
    test_dataset  = CIFAR10HDataset(images, probs, test_idx, transform=eval_transform)

    generator = _build_generator()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
