"""
=============================================================================
Fish Species Detection and Health Classification System
=============================================================================
Script  : train_fish_species.py
Author  : Senior ML Engineer
Purpose : Transfer-learning pipeline to classify fish species using ResNet50
Classes : pomfret, mackerel, black_snapper, prawn, pink_perch, black_pomfret

Usage:
    python train_fish_species.py

Requirements:
    pip install torch torchvision tqdm
=============================================================================
"""

import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


# =============================================================================
# 1. CONFIGURATION
#    All hyper-parameters and paths are centralised here so the script is easy
#    to tweak without hunting through the code.
# =============================================================================

CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    "data_dir":        "fish_project/datasets/species_clean",   # root of train/val/test
    "model_save_path": "models/fish_species_model.pth",

    # ── Classes (must match the sub-folder names exactly) ──────────────────
    "class_names": [
        "black_pomfret",
        "black_snapper",
        "mackerel",
        "pink_perch",
        "pomfret",
        "prawn",
    ],

    # ── Training hyper-parameters ──────────────────────────────────────────
    "batch_size":     32,
    "epochs":         15,
    "learning_rate":  0.0001,
    "num_workers":    4,        # parallel DataLoader workers
    "image_size":     224,      # ResNet50 standard input size

    # ── ImageNet normalisation stats (used by all torchvision pre-trained models) ──
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225],
}

NUM_CLASSES = len(CONFIG["class_names"])


# =============================================================================
# 2. DEVICE SETUP
#    Automatically use GPU (CUDA / Apple MPS) when available.
# =============================================================================

def get_device() -> torch.device:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Apple MPS detected.")
    else:
        device = torch.device("cpu")
        print("[Device] No GPU found – running on CPU (training will be slow).")
    return device


# =============================================================================
# 3. DATA TRANSFORMS
#    Training  → aggressive augmentation to improve generalisation.
#    Validation/Test → deterministic resize + normalise only.
# =============================================================================

def get_transforms(cfg: dict) -> dict:
    """
    Build torchvision transform pipelines for train, val, and test splits.

    Returns
    -------
    dict with keys 'train', 'val', 'test'
    """
    img_size = cfg["image_size"]
    mean     = cfg["mean"]
    std      = cfg["std"]

    train_transforms = transforms.Compose([
        # ── Spatial augmentations ──────────────────────────────────────────
        transforms.Resize((img_size + 32, img_size + 32)),   # slightly larger …
        transforms.RandomCrop(img_size),                      # … then random crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),

        # ── Colour augmentations ───────────────────────────────────────────
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
        ),

        # ── Tensor conversion + ImageNet normalisation ─────────────────────
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return {
        "train": train_transforms,
        "val":   eval_transforms,
        "test":  eval_transforms,
    }


# =============================================================================
# 4. DATASET & DATALOADER
#    Uses torchvision.datasets.ImageFolder which expects the directory layout:
#       <split>/<class_name>/<image_file>
# =============================================================================

def load_datasets(cfg: dict, data_transforms: dict) -> tuple:
    """
    Load train / val / test ImageFolder datasets and wrap them in DataLoaders.

    Returns
    -------
    (datasets_dict, dataloaders_dict, dataset_sizes_dict)
    """
    data_dir    = cfg["data_dir"]
    batch_size  = cfg["batch_size"]
    num_workers = cfg["num_workers"]

    splits = ["train", "val", "test"]

    image_datasets = {
        split: datasets.ImageFolder(
            root=os.path.join(data_dir, split),
            transform=data_transforms[split],
        )
        for split in splits
    }

    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            shuffle=True,               # shuffle every epoch during training
            num_workers=num_workers,
            pin_memory=True,            # faster CPU→GPU transfer
        ),
        "val": DataLoader(
            image_datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            image_datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    dataset_sizes = {split: len(image_datasets[split]) for split in splits}

    # ── Sanity-check: verify folders match expected classes ─────────────────
    detected_classes = image_datasets["train"].classes
    print(f"\n[Dataset] Detected classes : {detected_classes}")
    print(f"[Dataset] Expected classes : {cfg['class_names']}")
    if sorted(detected_classes) != sorted(cfg["class_names"]):
        print(
            "[WARNING] Detected class list does not match CONFIG['class_names']. "
            "Verify your folder names."
        )

    for split in splits:
        print(
            f"[Dataset] {split:>5} split → {dataset_sizes[split]:>5} images "
            f"({len(dataloaders[split])} batches)"
        )

    return image_datasets, dataloaders, dataset_sizes


# =============================================================================
# 5. MODEL DEFINITION
#    ResNet50 with ImageNet pre-trained weights.
#    Only the final fully-connected layer is replaced to match NUM_CLASSES.
#    Two training strategies are supported:
#      • feature_extract=True  → freeze all layers except the new FC head
#      • feature_extract=False → fine-tune the entire network (slower but better)
# =============================================================================

def build_model(num_classes: int, feature_extract: bool = False) -> nn.Module:
    """
    Build a ResNet50 model adapted for fish species classification.

    Parameters
    ----------
    num_classes     : number of output classes
    feature_extract : if True, freeze backbone weights (faster training)

    Returns
    -------
    torch.nn.Module
    """
    # Load pre-trained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # ── Optionally freeze all backbone layers ──────────────────────────────
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
        print("[Model] Backbone frozen – only FC head will be trained.")
    else:
        print("[Model] Full fine-tuning enabled – all layers will be trained.")

    # ── Replace the final fully-connected layer ────────────────────────────
    #    ResNet50's original FC: Linear(2048 → 1000)
    #    We replace it with:    Linear(2048 → num_classes)
    in_features = model.fc.in_features          # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),                      # regularisation
        nn.Linear(in_features, num_classes),
    )

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] ResNet50 loaded  | Total params: {total_params:,}")
    print(f"[Model] Trainable params : {trainable_params:,}")

    return model


# =============================================================================
# 6. TRAINING LOOP
#    One epoch of forward pass + back-propagation on the training split.
# =============================================================================

def train_one_epoch(
    model:       nn.Module,
    dataloader:  DataLoader,
    criterion:   nn.Module,
    optimizer:   optim.Optimizer,
    device:      torch.device,
    epoch:       int,
    total_epochs: int,
) -> tuple:
    """
    Train for a single epoch.

    Returns
    -------
    (epoch_loss, epoch_accuracy)
    """
    model.train()   # activates dropout, batch-norm in training mode

    running_loss    = 0.0
    running_correct = 0
    total_samples   = 0

    # tqdm progress bar for the training batches
    loop = tqdm(
        dataloader,
        desc=f"  Train [{epoch:02d}/{total_epochs}]",
        leave=False,
        dynamic_ncols=True,
    )

    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── Forward pass ───────────────────────────────────────────────────
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)

        # ── Backward pass + weight update ──────────────────────────────────
        loss.backward()
        optimizer.step()

        # ── Accumulate metrics ─────────────────────────────────────────────
        batch_size       = images.size(0)
        running_loss    += loss.item() * batch_size
        preds            = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total_samples   += batch_size

        # live loss on the progress bar
        loop.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss    / total_samples
    epoch_acc  = running_correct / total_samples

    return epoch_loss, epoch_acc


# =============================================================================
# 7. VALIDATION LOOP
#    Evaluate model performance on the validation split (no gradient updates).
# =============================================================================

def validate(
    model:      nn.Module,
    dataloader: DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
    epoch:      int,
    total_epochs: int,
) -> tuple:
    """
    Evaluate on the validation set.

    Returns
    -------
    (val_loss, val_accuracy)
    """
    model.eval()    # deactivates dropout, uses running stats for batch-norm

    running_loss    = 0.0
    running_correct = 0
    total_samples   = 0

    loop = tqdm(
        dataloader,
        desc=f"    Val [{epoch:02d}/{total_epochs}]",
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():   # disable gradient computation for speed & memory
        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            batch_size       = images.size(0)
            running_loss    += loss.item() * batch_size
            preds            = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total_samples   += batch_size

            loop.set_postfix(loss=f"{loss.item():.4f}")

    val_loss = running_loss    / total_samples
    val_acc  = running_correct / total_samples

    return val_loss, val_acc


# =============================================================================
# 8. FULL TRAINING PIPELINE
#    Orchestrates: epoch loop → train → validate → scheduler → checkpoint.
# =============================================================================

def train_model(
    model:        nn.Module,
    dataloaders:  dict,
    dataset_sizes: dict,
    cfg:          dict,
    device:       torch.device,
) -> nn.Module:
    """
    Full training pipeline with best-model checkpointing.

    Returns
    -------
    model with the best validation accuracy weights loaded
    """
    epochs        = cfg["epochs"]
    lr            = cfg["learning_rate"]
    save_path     = cfg["model_save_path"]

    # ── Loss function ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # ── Optimiser: Adam with weight decay for regularisation ───────────────
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    # ── LR Scheduler: reduce LR by ×0.1 if val loss plateaus for 3 epochs ──
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3,
    )

    # ── Checkpoint tracking ────────────────────────────────────────────────
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc   = 0.0

    # ── History for plotting / analysis ───────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("\n" + "=" * 65)
    print("  Starting Training")
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  Batch: {cfg['batch_size']}")
    print("=" * 65)

    start_time = time.time()

    for epoch in range(1, epochs + 1):

        # ── Train ──────────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer,
            device, epoch, epochs,
        )

        # ── Validate ───────────────────────────────────────────────────────
        val_loss, val_acc = validate(
            model, dataloaders["val"], criterion,
            device, epoch, epochs,
        )

        # ── Scheduler step ─────────────────────────────────────────────────
        scheduler.step(val_loss)

        # ── Record history ─────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ── Console output ─────────────────────────────────────────────────
        marker = "  ★ NEW BEST" if val_acc > best_val_acc else ""
        print(
            f"Epoch [{epoch:02d}/{epochs}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:.2f}%  |  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc*100:.2f}%"
            f"{marker}"
        )

        # ── Save best model ────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

            # Save full checkpoint (weights + metadata)
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": best_model_wts,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc":          best_val_acc,
                    "val_loss":         val_loss,
                    "class_names":      cfg["class_names"],
                    "config":           cfg,
                },
                save_path,
            )

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"  Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"  Best Validation Accuracy : {best_val_acc * 100:.2f}%")
    print(f"  Model saved to           : {save_path}")
    print("=" * 65 + "\n")

    # ── Reload best weights before returning ──────────────────────────────
    model.load_state_dict(best_model_wts)

    return model, history


# =============================================================================
# 9. TEST EVALUATION
#    Final accuracy on the held-out test split after training is complete.
# =============================================================================

def evaluate_on_test(
    model:      nn.Module,
    dataloader: DataLoader,
    device:     torch.device,
) -> float:
    """
    Report final accuracy on the test split.

    Returns
    -------
    test accuracy (float, 0–1)
    """
    model.eval()

    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="  Test Eval", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    test_acc = correct / total
    print(f"\n[Test] Accuracy on test split: {test_acc * 100:.2f}%  ({correct}/{total})")
    return test_acc


# =============================================================================
# 10. MAIN ENTRY POINT
# =============================================================================

def main():
    # ── Setup ──────────────────────────────────────────────────────────────
    device = get_device()

    # ── Transforms ─────────────────────────────────────────────────────────
    data_transforms = get_transforms(CONFIG)

    # ── Datasets & DataLoaders ─────────────────────────────────────────────
    image_datasets, dataloaders, dataset_sizes = load_datasets(
        CONFIG, data_transforms
    )

    # ── Model ──────────────────────────────────────────────────────────────
    #   feature_extract=False → fine-tune all layers (recommended for small datasets)
    #   feature_extract=True  → train only the final FC head (faster, less overfit)
    model = build_model(num_classes=NUM_CLASSES, feature_extract=False)
    model = model.to(device)

    # ── Train ──────────────────────────────────────────────────────────────
    trained_model, history = train_model(
        model, dataloaders, dataset_sizes, CONFIG, device
    )

    # ── Final Test Evaluation ──────────────────────────────────────────────
    evaluate_on_test(trained_model, dataloaders["test"], device)

    print("\n[Done] All steps complete. Model checkpoint saved to:")
    print(f"       {CONFIG['model_save_path']}\n")


if __name__ == "__main__":
    main()
