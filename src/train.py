import os
import pickle
import warnings

import numpy as np
import torch

from config import BATCH_SIZE, EARLY_STOPPING_PATIENCE, EPOCHS, LR, SEED
from src.dataloader import get_dataloaders
from src.losses import js_divergence_loss, kl_divergence_loss
from src.model import DissagreementPredictor
from src.utils import set_seed

PRETRAINED_BACKBONE_PATH = "checkpoints/backbone_cifar10_pretrained.pt"


def load_cifar_batch(batch_path):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
            category=np.exceptions.VisibleDeprecationWarning,
        )
        with open(batch_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")

    return batch[b"data"]


def train_model(model, train_loader, val_loader, loss_fn, device, save_path="checkpoints/best_model.pt"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_jsd": [],
    }

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        val_jsds = []

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)

                outputs = model(imgs)
                val_losses.append(loss_fn(outputs, targets).item())
                val_jsds.append(js_divergence_loss(outputs, targets).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_jsd = float(np.mean(val_jsds))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_jsd"].append(val_jsd)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(
            f"Epoch {epoch + 1}: "
            f"train={train_loss:.4f}, val={val_loss:.4f}, val_jsd={val_jsd:.4f}"
        )

    return history


def load_pretrained_backbone(model, backbone_path=PRETRAINED_BACKBONE_PATH, device="cpu"):
    if not os.path.exists(backbone_path):
        raise FileNotFoundError(
            f"Missing pretrained backbone at '{backbone_path}'. "
            "Run `python -m src.pretrain_cifar10` first."
        )

    state_dict = torch.load(backbone_path, map_location=device)
    model.backbone.load_state_dict(state_dict)


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    images = load_cifar_batch("data/raw/cifar-10-batches-py/test_batch")
    probs = np.load("data/raw/cifar10h-probs.npy")
    train_loader, val_loader, _ = get_dataloaders(images, probs, batch_size=BATCH_SIZE)

    model = DissagreementPredictor().to(device)
    load_pretrained_backbone(model, device=device)
    train_model(model, train_loader, val_loader, kl_divergence_loss, device)

    print("Training complete!")


if __name__ == "__main__":
    main()
