# Predicting Human Annotator Disagreement on CIFAR-10H

This project studies **disagreement prediction** for image classification using **CIFAR-10H**, a soft-label extension of CIFAR-10 that captures how human annotators distribute their votes across classes. Instead of predicting a single hard label, the model predicts a full probability distribution over the 10 CIFAR classes, with a focus on modeling human uncertainty.

## Problem

Standard classifiers collapse supervision to one-hot labels and discard ambiguity. This repository reframes the task as **soft-label prediction**: given a CIFAR-10 image, predict the human annotator label distribution and its associated uncertainty.

## Dataset

- **Primary dataset:** CIFAR-10H
- **Purpose:** soft labels for the 10,000-image CIFAR-10 test set
- **Scale:** approximately 51 human annotations per image
- **Source:** https://github.com/jcpeterson/cifar-10h

The project also uses the original CIFAR-10 training set for backbone pretraining.

## Method Overview

- **Backbone:** ResNet-18 adapted for 32x32 inputs
- **Prediction target:** 10-way soft label distribution
- **Heads explored:** linear, MLP, and temperature-scaled variants
- **Losses compared:**
  - KL divergence
  - Jensen-Shannon divergence (JSD)
  - Custom entropy-calibrated KL loss

## Experiments

- **Core training:** soft-label prediction on CIFAR-10H
- **Ablations:** backbone initialization and prediction head design
- **Robustness:** entropy correlation, corruption sensitivity, and class-conditional behavior
- **Explainability:** Grad-CAM on low-entropy and high-entropy examples

## Results Summary

| Loss | KL ↓ | JSD ↓ | Cosine ↑ | Pearson ↑ | Spearman ↑ | P@100 ↑ | P@200 ↑ | P@500 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KL Divergence | 0.2871 | 0.0595 | 0.9248 | 0.4057 | 0.3626 | 0.22 | 0.275 | 0.500 |
| Jensen-Shannon | 0.3519 | 0.0685 | 0.9097 | 0.3886 | 0.3274 | 0.22 | 0.320 | 0.462 |
| Custom (KL + Entropy) | 0.3095 | **0.0567** | **0.9263** | **0.4205** | **0.3949** | **0.23** | **0.340** | **0.504** |

## Key Findings

- The custom entropy-calibrated KL loss gave the best overall distribution-matching and entropy-prediction performance.
- CIFAR-10 pretraining improved downstream disagreement prediction.
- The MLP head outperformed simpler alternatives in the ablation study.
- Grad-CAM visualizations suggest diffuse attention on ambiguous examples and more localized attention on confident examples.

## Repository Structure

```text
.
├── config.py
├── data/
│   ├── processed/
│   └── raw/
├── figures/
├── notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_evaluation.ipynb
│   ├── 03_ablation.ipynb
│   ├── 04_robustness.ipynb
│   └── 05_explainability.ipynb
├── project_info/
├── reports/
│   ├── archive/
│   ├── final/
│   └── final_report.pdf
├── src/
│   ├── dataloader.py
│   ├── dataset.py
│   ├── evaluation.py
│   ├── losses.py
│   ├── model.py
│   ├── pretrain_cifar10.py
│   ├── train.py
│   ├── utils.py
│   └── misc/
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## How To Run

Use either the notebooks for experiment walkthroughs or the Python entrypoints directly.

```bash
python -m src.pretrain_cifar10
python -m src.train
```

For analysis and visualization, use:

```bash
jupyter notebook
```

## Reports and Figures

- Final submission PDF: [reports/final_report.pdf](reports/final_report.pdf)
- Final LaTeX source bundle: [reports/final/](reports/final/)
- Figures for training, ablation, robustness, and Grad-CAM are collected under [figures/](figures/)
