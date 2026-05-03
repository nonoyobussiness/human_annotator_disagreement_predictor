import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10

from config import BATCH_SIZE, EARLY_STOPPING_PATIENCE, EPOCHS, LR, SEED
from src.dataset import eval_transform, train_transform
from src.model import DissagreementPredictor
from src.utils import set_seed


PRETRAINED_BACKBONE_PATH = "checkpoints/backbone_cifar10_pretrained.pt"
TRAIN_SIZE = 45000
VAL_SIZE = 5000


def build_cifar10_resnet18(num_classes=10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_pretrain_dataloaders(batch_size=BATCH_SIZE):
    full_train = CIFAR10(root="data/raw", train=True, download=True, transform=train_transform)
    val_dataset = CIFAR10(root="data/raw", train=True, download=True, transform=eval_transform)
    test_dataset = CIFAR10(root="data/raw", train=False, download=True, transform=eval_transform)

    split_generator = torch.Generator().manual_seed(SEED)
    train_subset, val_subset_with_train_transform = random_split(
        full_train, [TRAIN_SIZE, VAL_SIZE], generator=split_generator
    )
    val_subset = Subset(val_dataset, val_subset_with_train_transform.indices)

    loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, generator=loader_generator)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            logits = model(imgs)
            losses.append(loss_fn(logits, targets).item())
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return float(np.mean(losses)), correct / total


def save_backbone_weights(model, save_path=PRETRAINED_BACKBONE_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    backbone = nn.Sequential(*list(model.children())[:-1])
    torch.save(backbone.state_dict(), save_path)


def pretrain(model, train_loader, val_loader, device, save_path=PRETRAINED_BACKBONE_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        train_loss = float(np.mean(train_losses))
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(
            f"Epoch {epoch + 1}: "
            f"train={train_loss:.4f}, val={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    save_backbone_weights(model, save_path=save_path)


def verify_backbone_compatibility(backbone_path=PRETRAINED_BACKBONE_PATH, device="cpu"):
    model = DissagreementPredictor().to(device)
    state_dict = torch.load(backbone_path, map_location=device)
    model.backbone.load_state_dict(state_dict)


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_pretrain_dataloaders(batch_size=BATCH_SIZE)
    model = build_cifar10_resnet18().to(device)

    pretrain(model, train_loader, val_loader, device)
    verify_backbone_compatibility(device=device)

    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Saved backbone to {PRETRAINED_BACKBONE_PATH}")
    print(f"Test loss={test_loss:.4f}, test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
