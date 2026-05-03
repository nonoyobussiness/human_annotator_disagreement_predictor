# End-to-End Project Pipeline: Predicting Human Annotator Disagreement
### DNN 3rd Year Course Project — Complete Execution Guide

---

## 📋 Table of Contents
1. [Environment Setup](#phase-0)
2. [Data Pipeline](#phase-1)
3. [Model Architecture](#phase-2)
4. [Loss Function Design](#phase-3)
5. [Training Protocol](#phase-4)
6. [Core Experiments & Evaluation](#phase-5)
7. [Ablation Studies](#phase-6)
8. [Robustness Checks](#phase-7)
9. [Explainability & Analysis](#phase-8)
10. [Report Writing](#phase-9)
11. [Submission](#phase-10)
12. [Common Pitfalls & How to Avoid Them](#pitfalls)

---

## Phase 0 — Environment Setup {#phase-0}

### 0.1 Recommended Stack
```
Python        3.10+
PyTorch       2.x (with CUDA support if GPU available)
torchvision   0.x (matching PyTorch version)
numpy         1.24+
matplotlib    3.7+
seaborn       0.12+
scikit-learn  1.3+
scipy         1.10+
grad-cam      1.4+ (pytorch-grad-cam)
tqdm          4.65+
wandb         OR tensorboard (for logging)
jupyter       (for EDA and visualization notebooks)
```

### 0.2 Project Directory Structure
```
project/
├── data/
│   ├── raw/                    # Downloaded CIFAR-10 and CIFAR-10H
│   └── processed/              # Aligned, split, cleaned data
├── src/
│   ├── dataset.py              # Dataset class and data pipeline
│   ├── model.py                # Architecture definitions
│   ├── losses.py               # All loss function implementations
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Evaluation metrics
│   ├── ablations.py            # Ablation experiment runners
│   ├── robustness.py           # Robustness check runners
│   └── explainability.py       # Grad-CAM and failure analysis
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_model_dev.ipynb      # Model development scratchpad
│   └── 03_results_viz.ipynb    # Final result visualizations
├── checkpoints/                # Saved model weights
├── logs/                       # Training logs
├── figures/                    # All saved plots for report
├── results/                    # CSVs / JSONs of metric outputs
├── report/                     # Final report LaTeX or Word files
├── config.py                   # All hyperparameters in one place
├── requirements.txt
└── README.md
```

### 0.3 Reproducibility Setup
```python
# config.py — put ALL hyperparameters here, never hardcode
SEED = 42
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 150
EARLY_STOPPING_PATIENCE = 15    
CIFAR10H_TRAIN_SIZE = 6000
CIFAR10H_VAL_SIZE = 2000
CIFAR10H_TEST_SIZE = 2000
```

```python
# At the top of every script
import torch, numpy as np, random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

> ⚠️ **Critical:** Call `set_seed()` before any data splitting, model initialization, or training. Document your seed in the report.

---

## Phase 1 — Data Pipeline {#phase-1}

### 1.1 Download the Datasets

**CIFAR-10** (images):
```python
import torchvision.datasets as datasets
cifar10_train = datasets.CIFAR10(root='data/raw', train=True, download=True)
cifar10_test  = datasets.CIFAR10(root='data/raw', train=False, download=True)
# cifar10_test images are the 10,000 images CIFAR-10H labels correspond to
```

**CIFAR-10H** (soft labels):
```bash
# Download from the official repository
wget https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-raw.zip
unzip cifar10h-raw.zip -d data/raw/
# This gives cifar10h-probs.npy — the 10,000 x 10 probability matrix
```

### 1.2 Alignment and Sanity Checks

```python
import numpy as np

cifar10h_probs = np.load('data/raw/cifar10h-probs.npy')  # shape: (10000, 10)

# Sanity Check 1: Shape
assert cifar10h_probs.shape == (10000, 10), "Shape mismatch!"

# Sanity Check 2: All distributions sum to 1
row_sums = cifar10h_probs.sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-5), "Distributions don't sum to 1!"

# Sanity Check 3: No NaN or negative values
assert not np.isnan(cifar10h_probs).any(), "NaN values found!"
assert (cifar10h_probs >= 0).all(), "Negative probabilities found!"

# Alignment: CIFAR-10H index i corresponds to CIFAR10 test image i
# The ordering is identical — no shuffle needed
```

> ⚠️ **Common pitfall:** CIFAR-10H corresponds to the CIFAR-10 **test** set (10,000 images), not the training set. Never mix these up.

### 1.3 Dataset Split

```python
from sklearn.model_selection import train_test_split

indices = np.arange(10000)

# First split off 2000 test images
train_val_idx, test_idx = train_test_split(
    indices, test_size=2000, random_state=SEED
)
# Then split remaining 8000 into 6000 train and 2000 val
train_idx, val_idx = train_test_split(
    train_val_idx, test_size=2000, random_state=SEED
)

# Save splits for reproducibility
np.save('data/processed/train_idx.npy', train_idx)
np.save('data/processed/val_idx.npy', val_idx)
np.save('data/processed/test_idx.npy', test_idx)
```

### 1.4 PyTorch Dataset Class

```python
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CIFAR10HDataset(Dataset):
    def __init__(self, cifar10_data, soft_labels, indices, transform=None):
        self.images = [cifar10_data[i][0] for i in indices]  # PIL images
        self.soft_labels = soft_labels[indices]               # (N, 10) float
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        return img, label

# Transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
```

> ⚠️ **Do NOT** use augmentations like color jitter, rotation, or cutout — these may distort the visual ambiguity that makes images disagreement-prone.

### 1.5 Required EDA Visualizations

**Plot 1 — Entropy Histogram:**
```python
from scipy.stats import entropy

true_entropies = np.array([entropy(p, base=2) for p in cifar10h_probs])

plt.figure(figsize=(8, 4))
plt.hist(true_entropies, bins=50, color='steelblue', edgecolor='white')
plt.xlabel('Shannon Entropy (bits)')
plt.ylabel('Number of Images')
plt.title('Distribution of Human Annotator Entropy across CIFAR-10H')
plt.savefig('figures/entropy_histogram.png', dpi=150, bbox_inches='tight')
```

**Plot 2 — Per-class Average Entropy:**
```python
classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']
hard_labels = np.array([cifar10_test[i][1] for i in range(10000)])

per_class_entropy = [true_entropies[hard_labels == c].mean() for c in range(10)]
plt.bar(classes, per_class_entropy)
plt.title('Mean Annotator Entropy per Class')
plt.savefig('figures/per_class_entropy.png', dpi=150, bbox_inches='tight')
```

**Plot 3 — Confusion-style Matrix:**
```python
import seaborn as sns

confusion = np.zeros((10, 10))
for probs, true_label in zip(cifar10h_probs, hard_labels):
    confusion[true_label] += probs
confusion = confusion / confusion.sum(axis=1, keepdims=True)

sns.heatmap(confusion, xticklabels=classes, yticklabels=classes,
            annot=True, fmt='.2f', cmap='Blues')
plt.title('Average Annotator Distribution per True Class')
plt.savefig('figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
```

**Plot 4 — Example Grid (Low vs High Entropy):**
```python
# Get top 8 lowest and highest entropy images
low_idx  = np.argsort(true_entropies)[:8]
high_idx = np.argsort(true_entropies)[-8:]
# Plot grid with image + bar chart of distribution beneath each
```

---

## Phase 2 — Model Architecture {#phase-2}

### 2.1 Backbone — Recommended: ResNet-18 (adapted for 32×32)

```python
import torchvision.models as models
import torch.nn as nn

class DisagreementPredictor(nn.Module):
    def __init__(self, backbone='resnet18', head='linear',
                 pretrained='cifar10', num_classes=10):
        super().__init__()

        # --- Backbone ---
        base = models.resnet18(pretrained=(pretrained == 'imagenet'))

        # CRITICAL: Adapt for CIFAR-10 32x32 images
        # Default ResNet uses 7x7 conv with stride 2 and maxpool
        # This crushes 32x32 images too aggressively
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()  # Remove maxpool for small images

        self.backbone = nn.Sequential(*list(base.children())[:-1])
        feat_dim = 512  # ResNet-18 output features

        # --- Prediction Head ---
        if head == 'linear':
            self.head = nn.Linear(feat_dim, num_classes)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        elif head == 'temperature':
            self.head = nn.Linear(feat_dim, num_classes)
            self.temperature = nn.Parameter(torch.ones(1))

        self.head_type = head

    def forward(self, x):
        feat = self.backbone(x).squeeze(-1).squeeze(-1)
        logits = self.head(feat)
        if self.head_type == 'temperature':
            logits = logits / self.temperature.clamp(min=0.1)
        return torch.softmax(logits, dim=-1)
```

> ⚠️ **Critical pitfall:** If you use the default ResNet-18 from torchvision without modifying conv1 and removing maxpool, the 32×32 images will be spatially crushed to 1×1 before the FC layer. Always make this adaptation.

### 2.2 Backbone Initialization Options
You need to compare **three** strategies:

| Strategy | How to implement |
|---|---|
| Random init | `pretrained=False`, no loading |
| ImageNet pretrained | `pretrained=True` from torchvision |
| CIFAR-10 hard-label pretrained | Train on 50k hard labels first, save weights, load here |

For CIFAR-10 pretraining:
```python
# Step 1: Pretrain on CIFAR-10 hard labels (standard cross-entropy)
# Step 2: Save backbone weights
torch.save(model.backbone.state_dict(), 'checkpoints/backbone_cifar10_pretrained.pt')
# Step 3: Load in your soft-label model
model.backbone.load_state_dict(torch.load('checkpoints/backbone_cifar10_pretrained.pt'))
```

### 2.3 Required: Architecture Diagram
Draw (or generate) a diagram showing:
- Input (32×32×3) → Adapted Conv1 → ResNet blocks → Global Avg Pool → Feature Vector → Head → Softmax → 10-dim distribution

Save as `figures/architecture_diagram.png`

### 2.4 Parameter Count Table

```python
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# Report this for each model variant in a table
```

---

## Phase 3 — Loss Function Design {#phase-3}

### 3.1 Loss 1 (Mandatory) — KL Divergence

```python
import torch.nn.functional as F

def kl_divergence_loss(q, p, eps=1e-8):
    """
    KL(p || q): how much q diverges from the true distribution p.
    q = predicted distribution (model output, already softmaxed)
    p = true soft label distribution
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p.log() - q.log())).sum(dim=-1).mean()
```

**Report justification:** KL divergence penalizes the model for placing low probability on classes that humans frequently chose. It is asymmetric — it heavily penalizes placing near-zero probability on classes with non-zero human votes.

### 3.2 Loss 2 (Mandatory) — Jensen-Shannon Divergence

```python
def js_divergence_loss(q, p, eps=1e-8):
    """
    JSD is symmetric and bounded [0, 1] when using log base 2.
    More stable than KL when distributions have zero entries.
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    jsd = 0.5 * (p * (p.log() - m.log())).sum(dim=-1) + \
          0.5 * (q * (q.log() - m.log())).sum(dim=-1)
    return jsd.mean()
```

**Report justification:** JSD is symmetric and numerically more stable. It avoids the infinite penalty problem of KL when q assigns zero to a class p assigns non-zero probability.

### 3.3 Loss 3 (Mandatory) — Custom Composite Loss

Design a meaningful loss. Recommended: **KL + Entropy Calibration Penalty**

```python
def entropy_calibrated_kl_loss(q, p, alpha=0.5, eps=1e-8):
    """
    Custom loss = KL divergence + penalty for getting entropy wrong.

    Intuition: Beyond matching the distribution shape, we want the model
    to correctly predict HOW MUCH humans disagree (entropy level).
    A model that predicts the right top class but completely wrong
    entropy profile should be penalized extra.

    alpha controls the trade-off between distribution matching
    and entropy calibration.
    """
    # Base KL loss
    kl = kl_divergence_loss(q, p, eps)

    # Entropy calibration term
    true_entropy  = -(p * (p + eps).log()).sum(dim=-1)   # H(p)
    pred_entropy  = -(q * (q + eps).log()).sum(dim=-1)   # H(q)
    entropy_error = F.mse_loss(pred_entropy, true_entropy)

    return kl + alpha * entropy_error
```

**Report justification:** Standard KL can match a distribution's shape while still being badly calibrated on entropy (e.g., the model spreads probability evenly when humans mostly agreed on one class). The entropy penalty explicitly forces the model to predict the *degree* of disagreement, not just its direction.

### 3.4 Loss 4 (Bonus) — Earth Mover's Distance

```python
from scipy.stats import wasserstein_distance

# Define a class distance matrix (semantic distances between CIFAR-10 classes)
# Example: cat-dog closer than cat-airplane
CLASS_DISTANCES = np.array([...])  # 10x10 matrix, justify in report

def emd_loss_batch(q_batch, p_batch):
    """Compute mean EMD over a batch (use scipy, move to numpy)."""
    q_np = q_batch.detach().cpu().numpy()
    p_np = p_batch.detach().cpu().numpy()
    emds = [wasserstein_distance(p_np[i], q_np[i]) for i in range(len(p_np))]
    # Convert back to tensor for autograd
    return torch.tensor(np.mean(emds), requires_grad=True)
```

> ⚠️ **Pitfall:** EMD via scipy is not differentiable in the PyTorch graph. Use it only as an **evaluation metric**, or use the `pot` (Python Optimal Transport) library for a differentiable version if training with it.

---

## Phase 4 — Training Protocol {#phase-4}

### 4.1 Training Loop

```python
def train_model(model, train_loader, val_loader, loss_fn,
                epochs=150, lr=1e-3, patience=15,
                save_path='checkpoints/best_model.pt'):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_jsd': []}

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_losses = []
        for imgs, soft_labels in train_loader:
            imgs, soft_labels = imgs.to(DEVICE), soft_labels.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, soft_labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval()
        val_losses, val_jsds = [], []
        with torch.no_grad():
            for imgs, soft_labels in val_loader:
                imgs, soft_labels = imgs.to(DEVICE), soft_labels.to(DEVICE)
                preds = model(imgs)
                val_losses.append(loss_fn(preds, soft_labels).item())
                val_jsds.append(js_divergence_loss(preds, soft_labels).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        val_jsd    = np.mean(val_jsds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_jsd'].append(val_jsd)

        scheduler.step()

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train={train_loss:.4f}, "
                  f"val={val_loss:.4f}, val_jsd={val_jsd:.4f}")

    return history
```

### 4.2 Training Consistency Rules
When comparing loss functions, **only the loss function changes.** Everything else is identical:

| Component | Fixed value |
|---|---|
| Optimizer | Adam, lr=1e-3, weight_decay=1e-4 |
| LR schedule | CosineAnnealingLR |
| Batch size | 64 |
| Max epochs | 150 |
| Early stopping | patience=15 on val loss |
| Augmentation | HorizontalFlip + RandomCrop(32, padding=4) |
| Backbone | ResNet-18 adapted for 32×32 |
| Initialization | CIFAR-10 pretrained (for main runs) |

### 4.3 Required Training Visualizations

```python
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Loss Curves'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_jsd'], label='Val JSD', color='orange')
plt.xlabel('Epoch'); plt.ylabel('JSD')
plt.title('Validation JSD over Epochs')
plt.savefig('figures/training_curves.png', dpi=150, bbox_inches='tight')
```

---

## Phase 5 — Core Experiments & Evaluation {#phase-5}

### 5.1 Load Best Model and Evaluate on Test Set

```python
model.load_state_dict(torch.load('checkpoints/best_model.pt'))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for imgs, soft_labels in test_loader:
        preds = model(imgs.to(DEVICE))
        all_preds.append(preds.cpu().numpy())
        all_true.append(soft_labels.numpy())

all_preds = np.concatenate(all_preds)   # (2000, 10)
all_true  = np.concatenate(all_true)    # (2000, 10)
```

### 5.2 Metric 1 — Distribution Matching

```python
from scipy.spatial.distance import jensenshannon

def compute_distribution_metrics(preds, true, eps=1e-8):
    kl_divs, jsds, cosines = [], [], []
    for p, q in zip(true, preds):
        # KL(p || q)
        p_c, q_c = p.clip(eps), q.clip(eps)
        kl_divs.append(np.sum(p_c * np.log(p_c / q_c)))
        # JSD (scipy returns sqrt(JSD), so square it)
        jsds.append(jensenshannon(p, q)**2)
        # Cosine similarity
        cosines.append(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q) + eps))

    return {
        'KL_mean': np.mean(kl_divs),   'KL_std': np.std(kl_divs),
        'JSD_mean': np.mean(jsds),     'JSD_std': np.std(jsds),
        'Cosine_mean': np.mean(cosines),'Cosine_std': np.std(cosines),
    }
```

### 5.3 Metric 2 — Entropy Prediction Quality

```python
from scipy.stats import pearsonr, spearmanr

true_ent = np.array([entropy(p, base=2) for p in all_true])
pred_ent = np.array([entropy(q, base=2) for q in all_preds])

pearson_r,  _ = pearsonr(true_ent, pred_ent)
spearman_r, _ = spearmanr(true_ent, pred_ent)
print(f"Pearson r: {pearson_r:.4f}, Spearman r: {spearman_r:.4f}")
```

### 5.4 Metric 3 — Precision@K

```python
def precision_at_k(true_entropy, pred_entropy, k):
    """
    What fraction of the top-K most disagreement-prone images
    (by predicted entropy) are actually in the top-K by true entropy?
    """
    true_topk = set(np.argsort(true_entropy)[-k:])
    pred_topk = set(np.argsort(pred_entropy)[-k:])
    return len(true_topk & pred_topk) / k

for k in [100, 200, 500]:
    p_at_k = precision_at_k(true_ent, pred_ent, k)
    print(f"Precision@{k}: {p_at_k:.4f}")
```

### 5.5 Summary Comparison Table

Run all metrics for each loss function's best model and compile:

| Loss Function | KL↓ | JSD↓ | Cosine↑ | Pearson r↑ | Spearman r↑ | P@100↑ | P@200↑ | P@500↑ |
|---|---|---|---|---|---|---|---|---|
| KL Divergence | | | | | | | | |
| Jensen-Shannon | | | | | | | | |
| Custom (KL + Entropy) | | | | | | | | |
| (Bonus) EMD | | | | | | | | |

### 5.6 Required Visualizations

**Scatter Plot — Predicted vs True Entropy:**
```python
plt.figure(figsize=(6, 6))
plt.scatter(true_ent, pred_ent, alpha=0.3, s=10)
plt.plot([0, 3.32], [0, 3.32], 'r--', label='Perfect prediction')
plt.xlabel('True Entropy'); plt.ylabel('Predicted Entropy')
plt.title(f'Entropy Prediction (Pearson r = {pearson_r:.3f})')
plt.savefig('figures/entropy_scatter.png', dpi=150, bbox_inches='tight')
```

**Qualitative Examples Grid:**
Show 5 low-entropy, 5 medium-entropy, 5 high-entropy test images with:
- The image
- True distribution bar chart
- Predicted distribution bar chart

---

## Phase 6 — Ablation Studies (Pick Any 3) {#phase-6}

### Ablation A — Backbone Initialization

Train the **same model + same loss** three times, varying only initialization:

```python
configs = [
    {'init': 'random',    'pretrained': False},
    {'init': 'imagenet',  'pretrained': True},
    {'init': 'cifar10',   'weights': 'checkpoints/backbone_cifar10_pretrained.pt'},
]
# Record val JSD and test KL for each
```

Expected outcome: CIFAR-10 pretrained > ImageNet pretrained > Random. If this doesn't hold, analyze and explain why.

### Ablation B — Loss Function Comparison       

Already done in Phase 5 — just present it formally as an ablation with:
- Same model, same seed, same training setup
- One table + one grouped bar chart

### Ablation C — Training Data Strategy

```python
# Strategy 1: Train on soft labels only (6000 CIFAR-10H images)
# Strategy 2: Pretrain on 50k hard labels → Fine-tune on 6000 soft labels

# For strategy 2 fine-tuning, use a lower learning rate
# (e.g., 1e-4 instead of 1e-3) to avoid catastrophic forgetting
```

> ⚠️ **Catastrophic forgetting:** When fine-tuning on soft labels after hard-label pretraining, use a small LR and consider freezing the backbone's early layers. If your val JSD worsens after fine-tuning starts, lower your LR.

### Ablation D — Prediction Head Architecture

Compare at minimum:
- Single linear layer + softmax
- MLP (2-layer) + softmax
- Temperature-scaled linear + softmax

Keep everything else identical.

### Ablation Visualization

```python
# One grouped bar chart showing your primary metric (e.g., JSD) across ablation conditions
# x-axis: conditions, y-axis: mean JSD on test set
```

---

## Phase 7 — Robustness Checks (Pick Any 2) {#phase-7}

### Check A — Annotator Subsampling

```python
# CIFAR-10H raw file has per-annotator responses
cifar10h_raw = np.load('data/raw/cifar10h-raw.npy')  # shape (10000, N_annotators)

# Subsample to k annotators and recompute soft distributions
for k in [5, 10, 20, 35, 51]:
    subsampled_probs = []
    for img_responses in cifar10h_raw:
        valid = img_responses[img_responses >= 0]  # remove padding -1s
        chosen = np.random.choice(valid, size=min(k, len(valid)), replace=False)
        dist = np.bincount(chosen.astype(int), minlength=10) / len(chosen)
        subsampled_probs.append(dist)
    subsampled_probs = np.array(subsampled_probs)
    # Compute JSD between subsampled and full distribution
    # and between model predictions and subsampled distribution
```

### Check B — OOD Corruptions

```python
from torchvision.transforms import functional as TF
import cv2

def apply_corruption(img_tensor, corruption_type, severity):
    """severity: 1 (mild) to 5 (severe)"""
    img_np = img_tensor.permute(1,2,0).numpy()

    if corruption_type == 'gaussian_noise':
        noise_std = [0.04, 0.08, 0.12, 0.18, 0.26][severity-1]
        img_np = img_np + np.random.randn(*img_np.shape) * noise_std

    elif corruption_type == 'gaussian_blur':
        kernel_size = [3, 5, 7, 9, 11][severity-1]
        img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)

    elif corruption_type == 'contrast':
        factor = [0.8, 0.6, 0.4, 0.2, 0.1][severity-1]
        img_np = img_np * factor + img_np.mean() * (1 - factor)

    return torch.tensor(img_np.clip(0,1)).permute(2,0,1).float()

# For each corruption type and severity, measure mean predicted entropy
# Plot: severity (x) vs mean predicted entropy (y)
# Expected: entropy should increase with severity (more corrupted = more uncertain)
```

> ⚠️ **Important:** If entropy does **not** increase with corruption severity, do **not** force that conclusion. Instead, analyze: Is the model overconfident? Is the backbone ignoring noise due to BatchNorm? Report and explain honestly.

### Check C — Class-conditional Performance

```python
# For each of the 10 CIFAR-10 classes, compute KL and JSD separately
hard_labels_test = np.array([cifar10_test[i][1] for i in test_idx])

for cls in range(10):
    cls_mask = hard_labels_test == cls
    cls_preds = all_preds[cls_mask]
    cls_true  = all_true[cls_mask]
    metrics   = compute_distribution_metrics(cls_preds, cls_true)
    print(f"{classes[cls]}: KL={metrics['KL_mean']:.4f}, JSD={metrics['JSD_mean']:.4f}")

# Plot as horizontal bar chart per class
```

---

## Phase 8 — Explainability & Analysis {#phase-8}

### 8.1 Grad-CAM Analysis

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Target the last conv layer
cam = GradCAM(model=model, target_layers=[model.backbone[-1][-1]])

# Select examples: 5 low-entropy, 5 high-entropy test images
low_e_idx  = np.argsort(true_ent)[:5]
high_e_idx = np.argsort(true_ent)[-5:]

for idx in list(low_e_idx) + list(high_e_idx):
    input_tensor = test_dataset[idx][0].unsqueeze(0).to(DEVICE)
    grayscale_cam = cam(input_tensor=input_tensor)
    visualization = show_cam_on_image(raw_img_normalized, grayscale_cam[0])
    # Save and include in report
```

**What to discuss in report:**
- Low-entropy images: Does the model focus on a clear, central object?
- High-entropy images: Is attention diffuse, on background, or on ambiguous regions?
- Are there cases where the model attends to the right region but still predicts wrong entropy?

### 8.2 Failure Case Analysis

```python
# Compute per-image KL divergence between prediction and truth
per_image_kl = np.array([
    kl_divergence_loss_np(all_preds[i], all_true[i]) for i in range(len(all_true))
])

# Top 10 worst predictions
worst_idx = np.argsort(per_image_kl)[-10:]

for idx in worst_idx:
    # Show: image, true distribution, predicted distribution,
    # true entropy, predicted entropy, brief hypothesis
```

**For each failure case, hypothesize:**
- Did the model underestimate uncertainty on a genuinely ambiguous image?
- Did the model overestimate uncertainty on a clear image?
- Is the image at a class boundary?
- Is there class imbalance in this region of the distribution?

### 8.3 Manual Disagreement Source Analysis

Inspect the **top 30–50 highest-entropy images** manually (open them one by one). For each, assign a category:

| Category | Description |
|---|---|
| **Ambiguous identity** | The object genuinely looks like multiple classes |
| **Poor image quality** | Blurry, dark, low contrast |
| **Multi-object** | Two or more CIFAR classes visible |
| **Boundary case** | Object is at the edge between two semantically close classes |
| **Other** | Any other reason |

> ⚠️ **Viva requirement:** Every team member must do this personally for at least 5–10 images. You will be asked about specific images during the viva. Do not delegate this entirely to one person.

Create a summary table and pie chart of category distribution.

---

## Phase 9 — Report Writing {#phase-9}

### 9.1 Recommended Report Structure

```
1. Abstract (200 words)
2. Introduction
   - Problem statement
   - Why disagreement matters (cite Peterson et al. 2019)
   - Overview of contributions
3. Related Work
   - Soft labels, knowledge distillation, CIFAR-10H
4. Dataset
   - CIFAR-10H description
   - Split details
   - EDA visualizations
5. Methodology
   5.1 Data Pipeline
   5.2 Model Architecture (with diagram)
   5.3 Loss Functions (with math)
   5.4 Training Protocol
6. Experiments
   6.1 Core Evaluation (table + scatter + examples)
   6.2 Ablation Studies
   6.3 Robustness Checks
   6.4 Explainability Analysis
7. Discussion
   - What worked, what didn't, and why
   - Limitations
8. Conclusion
9. References
Appendix (extra figures, parameter tables)
```

### 9.2 Writing Rules
- Every figure must have a caption explaining what it shows and what conclusion to draw
- Every table must have a header row and a caption
- For every experiment: state what you did → what you expected → what happened → why
- Define every metric before using it
- Do not include code in the main report — only pseudocode if needed
- Cite Peterson et al. (ICCV 2019) for CIFAR-10H

### 9.3 Checklist Before Submitting Report

- [ ] Entropy histogram included
- [ ] Per-class entropy plot included
- [ ] Annotator confusion matrix included
- [ ] Low/high entropy example grid included
- [ ] Architecture diagram included
- [ ] Parameter count table included
- [ ] Training + validation loss curves for each main model
- [ ] Entropy scatter plot (predicted vs true) for best model
- [ ] Summary metrics table (all losses × all metrics)
- [ ] Ablation summary chart
- [ ] Robustness check plots
- [ ] Grad-CAM visualizations
- [ ] Failure case analysis (min. 5 cases)
- [ ] Manual disagreement source analysis table
- [ ] All random seeds documented
- [ ] Data split sizes documented
- [ ] All hyperparameters documented

---

## Phase 10 — Submission {#phase-10}

### 10.1 What to Submit
> ⚠️ The project document does not specify an exact submission portal. Confirm with your course instructor / TAs. Below is the standard format expected for this type of project.

**Deliverable 1 — Code Repository:**
- Clean, commented, runnable code
- `README.md` with: setup instructions, how to reproduce each experiment, random seeds used
- `requirements.txt` with pinned versions
- All model checkpoints (or a download link in README)
- All generated figures in `figures/`

**Deliverable 2 — Final Report:**
- PDF format (LaTeX preferred, Word acceptable)
- Recommended length: 8–12 pages (excluding references and appendix)
- All figures embedded, high resolution (min 150 DPI)
- Named: `[TeamName]_DNN_Project_Report.pdf`

**Deliverable 3 — Presentation / Viva:**
- Be ready to walk through: your architecture, any one experiment, the Grad-CAM results, and your manually inspected disagreement images
- Every team member must understand the full pipeline

### 10.2 Submission Format Summary

| Item | Format | Filename convention |
|---|---|---|
| Code | .zip or GitHub link | `TeamName_code.zip` |
| Report | PDF | `TeamName_DNN_Report.pdf` |
| Checkpoints | .pt files inside zip or Drive link | `best_model_[loss].pt` |
| Figures | .png inside code zip | `figures/` folder |

### 10.3 Final Pre-Submission Checklist

- [ ] All 3 compulsory ablations done
- [ ] All 2 compulsory robustness checks done
- [ ] All 3 explainability analyses done (Grad-CAM, failure cases, manual inspection)
- [ ] Summary metrics table complete for all loss functions
- [ ] Code runs from scratch with `README` instructions
- [ ] Seeds documented and fixed
- [ ] No hardcoded paths (use `config.py` or argparse)
- [ ] Report PDF compiles without errors
- [ ] Every team member has personally inspected high-entropy images for viva

---

## Common Pitfalls & How to Avoid Them {#pitfalls}

| Pitfall | What goes wrong | Fix |
|---|---|---|
| Using default ResNet-18 for 32×32 | Feature maps crushed to 1×1 by stride+maxpool | Replace conv1 with 3×3 stride-1, remove maxpool |
| Treating CIFAR-10 training labels as soft labels | Hard labels have no disagreement signal | Only use CIFAR-10 for pretraining, not as targets |
| Not fixing the random seed before train/val/test split | Results are not reproducible | Call `set_seed()` before any split |
| Using KL without clipping | log(0) = -inf, training explodes | Always clip probabilities: `p.clamp(min=1e-8)` |
| Comparing models trained with different seeds | Noise in results, unfair comparison | Use same seed for all ablation/comparison runs |
| Augmentations that change class identity | E.g. rotating a '9' to look like '6' | Stick to horizontal flip + crop only |
| Not saving best checkpoint during training | Last epoch may be worse than best | Always save on best val loss, not last epoch |
| Using EMD via scipy as a training loss | Not differentiable through PyTorch autograd | Use as evaluation metric only, or use `pot` library |
| Grad-CAM on wrong layer | Heatmaps are garbage | Target the last convolutional layer before global avg pool |
| Forgetting to normalize images for Grad-CAM overlay | Heatmap on wrong pixel range | Unnormalize image before calling `show_cam_on_image` |
| Only one team member does the manual analysis | All members fail viva questions | Split the high-entropy images across team members |

---

*Document prepared for: DNN 3rd Year Course Project — Predicting Human Annotator Disagreement*
*Based on: Peterson et al., "Human Uncertainty Makes Classification More Robust," ICCV 2019*