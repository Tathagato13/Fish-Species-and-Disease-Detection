# Fish Species and Disease Detection System

A deep learning system for **fish species classification** and **fish disease detection** using transfer learning with PyTorch.

This project implements two separate computer vision models trained on fish image datasets to assist in **aquaculture monitoring, fish farming automation, and marine species identification**.

---

# Project Overview

This repository contains **two deep learning pipelines**:

### 1. Fish Species Classification
Identifies the species of fish from an input image.

**Model Used**
- ResNet50 (ImageNet pretrained)

**Target Classes**

- black_pomfret  
- black_snapper  
- mackerel  
- pink_perch  
- pomfret  
- prawn  

---

### 2. Fish Disease Detection

Detects whether a fish is **Healthy or Diseased**.

**Model Used**

- EfficientNet-B0 (ImageNet pretrained)

**Target Classes**

- Healthy  
- Diseased  

---

# Project Structure
Fish-Species-and-Disease-Detection
│
├── models
│ ├── train_fish_species.py
│ ├── train_disease_model.py
│
├── datasets
│ ├── species_clean
│ │ ├── train
│ │ ├── val
│ │ └── test
│ │
│ └── disease_clean
│ ├── train
│ ├── val
│ └── test
│
├── requirements.txt
├── .gitignore
└── README.md

---

# Dataset Format

Both models use the **PyTorch ImageFolder format**.

Example structure:
datasets/

species_clean/
train/
black_pomfret/
black_snapper/
mackerel/
pink_perch/
pomfret/
prawn/

val/
    ...

test/
    ...
   disease_clean/
train/
healthy/
diseased/

val/
    ...

test/
    ...
    
---

# Model 1: Fish Species Classification

## Architecture
ResNet50 with transfer learning.

Pretrained on **ImageNet** and fine-tuned for fish classification.

### Training Parameters

| Parameter | Value |
|----------|------|
| Model | ResNet50 |
| Image Size | 224 x 224 |
| Batch Size | 32 |
| Epochs | 15 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |

### Training Command

```bash
python train_fish_species.py
