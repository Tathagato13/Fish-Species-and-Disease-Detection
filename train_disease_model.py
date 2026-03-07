"""
=============================================================================
  Fish Health Classification — Disease Detection Model
  Architecture : EfficientNet-B0 (ImageNet pretrained)
  Classes      : Healthy (0)  |  Diseased (1)
  Imbalance    : Handled via weighted CrossEntropyLoss
=============================================================================
"""

import os
import time
import copy
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# 0.  REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
# Paths
BASE_DIR    = Path(__file__).resolve().parent.parent   # project root
DATA_DIR    = BASE_DIR /"fish_project" / "datasets" / "disease_clean"
MODEL_DIR   = BASE_DIR / "models"
MODEL_PATH  = MODEL_DIR / "fish_health_model.pth"

# Hyperparameters
BATCH_SIZE  = 32
EPOCHS      = 15
LR          = 1e-4
IMG_SIZE    = 224
NUM_CLASSES = 2

# ImageNet normalisation stats (required for pretrained EfficientNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────────────────────────────────────
# 2.  DEVICE
# ──────────────────────────────────────────────────────────────────────────────
device = (
    torch.device("cuda")  if torch.cuda.is_available()  else
    torch.device("mps")   if torch.backends.mps.is_available() else
    torch.device("cpu")
)
print(f"\n{'='*60}")
print(f"  Fish Health Classification — Disease Detection")
print(f"{'='*60}")
print(f"  Device      : {device}")
print(f"  Data root   : {DATA_DIR}")
print(f"  Output model: {MODEL_PATH}")
print(f"{'='*60}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 3.  TRANSFORMS
# ──────────────────────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),          # slightly larger for crop
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),  # random scale crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ──────────────────────────────────────────────────────────────────────────────
# 4.  DATASETS & DATALOADERS
# ──────────────────────────────────────────────────────────────────────────────
print("Loading datasets …")

train_dataset = datasets.ImageFolder(
    root=str(DATA_DIR / "train"),
    transform=train_transforms
)
val_dataset = datasets.ImageFolder(
    root=str(DATA_DIR / "val"),
    transform=val_test_transforms
)
test_dataset = datasets.ImageFolder(
    root=str(DATA_DIR / "test"),
    transform=val_test_transforms
)

# Class index mapping (ImageFolder sorts alphabetically)
# → class_to_idx will be e.g. {'diseased': 0, 'healthy': 1}  (order may vary)
CLASS_NAMES = train_dataset.classes          # e.g. ['diseased', 'healthy']
print(f"  Class mapping : {train_dataset.class_to_idx}")
print(f"  Train samples : {len(train_dataset)}")
print(f"  Val   samples : {len(val_dataset)}")
print(f"  Test  samples : {len(test_dataset)}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 5.  CLASS WEIGHT CALCULATION  (inverse-frequency weighting)
# ──────────────────────────────────────────────────────────────────────────────
def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """
    Computes class weights as N_total / (N_classes * N_per_class).
    Returns a 1-D tensor of shape (num_classes,) on the target device.
    """
    labels      = np.array([label for _, label in dataset.samples])
    class_count = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    n_total     = len(labels)
    weights     = n_total / (NUM_CLASSES * class_count)   # inv-freq
    weights     = weights / weights.sum() * NUM_CLASSES   # keep sum ≈ N_classes
    print("  Per-class sample counts :", {
        CLASS_NAMES[i]: int(class_count[i]) for i in range(NUM_CLASSES)
    })
    print(f"  Class weights           : {weights.round(4)}\n")
    return torch.tensor(weights, dtype=torch.float32).to(device)

class_weights = compute_class_weights(train_dataset)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=(device.type == "cuda"),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=(device.type == "cuda"),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=(device.type == "cuda"),
)

# ──────────────────────────────────────────────────────────────────────────────
# 6.  MODEL — EfficientNet-B0 with custom head
# ──────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    """
    Load ImageNet-pretrained EfficientNet-B0 and replace the classifier head
    with a new Linear(in_features, num_classes) layer.
    All backbone weights remain trainable (full fine-tuning).
    """
    # torchvision >= 0.13 uses the 'weights' API
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = models.efficientnet_b0(weights=weights)

    # EfficientNet-B0 classifier: Sequential(Dropout, Linear(1280, 1000))
    in_features = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    print(f"  Backbone      : EfficientNet-B0 (ImageNet pretrained)")
    print(f"  Classifier    : Linear({in_features} → {num_classes})")
    print(f"  Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    return model

model = build_model(NUM_CLASSES).to(device)

# ──────────────────────────────────────────────────────────────────────────────
# 7.  LOSS, OPTIMISER, SCHEDULER
# ──────────────────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

# Cosine annealing: gradually reduces LR to near-zero over all epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6,
)

# ──────────────────────────────────────────────────────────────────────────────
# 8.  HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy for a batch."""
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}m {s:02d}s"


# ──────────────────────────────────────────────────────────────────────────────
# 9.  TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
) -> tuple[float, float]:
    """Run one full pass over the training set. Returns (avg_loss, avg_acc)."""
    model.train()
    running_loss = 0.0
    running_acc  = 0.0

    pbar = tqdm(
        loader,
        desc=f"  [Train] Epoch {epoch:02d}/{EPOCHS}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping — stabilises training with class-weighted loss
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_loss = loss.item()
        batch_acc  = accuracy(outputs, labels)
        running_loss += batch_loss
        running_acc  += batch_acc

        pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.3f}")

    n = len(loader)
    return running_loss / n, running_acc / n


# ──────────────────────────────────────────────────────────────────────────────
# 10.  VALIDATION LOOP
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    split: str = "Val",
) -> tuple[float, float]:
    """Run inference on val or test set. Returns (avg_loss, avg_acc)."""
    model.eval()
    running_loss = 0.0
    running_acc  = 0.0

    with torch.no_grad():
        pbar = tqdm(
            loader,
            desc=f"  [{split:5s}]       ",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            loss           = criterion(outputs, labels)
            running_loss  += loss.item()
            running_acc   += accuracy(outputs, labels)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy(outputs, labels):.3f}")

    n = len(loader)
    return running_loss / n, running_acc / n


# ──────────────────────────────────────────────────────────────────────────────
# 11.  MAIN TRAINING RUN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_val_acc   = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history        = []

    print(f"{'─'*60}")
    print(f"  Starting training  —  {EPOCHS} epochs")
    print(f"{'─'*60}\n")

    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )

        # ── Validate ────────────────────────────────────────────────────────
        val_loss, val_acc = evaluate(model, val_loader, criterion, split="Val")

        # ── LR Scheduler step ───────────────────────────────────────────────
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Checkpoint best model ───────────────────────────────────────────
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc   = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch"           : epoch,
                    "model_state_dict": best_model_wts,
                    "val_acc"         : best_val_acc,
                    "class_names"     : CLASS_NAMES,
                    "class_to_idx"    : train_dataset.class_to_idx,
                },
                MODEL_PATH,
            )

        epoch_time = time.time() - epoch_start

        # ── Console output ───────────────────────────────────────────────────
        marker = " ★ BEST" if is_best else ""
        print(
            f"  Epoch [{epoch:02d}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} "
            f"| LR: {current_lr:.2e} "
            f"| {format_time(epoch_time)}"
            f"{marker}"
        )

        history.append({
            "epoch"     : epoch,
            "train_loss": train_loss,
            "train_acc" : train_acc,
            "val_loss"  : val_loss,
            "val_acc"   : val_acc,
        })

    total_time = time.time() - total_start

    # ──────────────────────────────────────────────────────────────────────────
    # 12.  FINAL TEST EVALUATION  (on held-out test set)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Training complete in {format_time(total_time)}")
    print(f"  Best Val Accuracy : {best_val_acc:.4f}")
    print(f"  Model saved to    : {MODEL_PATH}")
    print(f"{'─'*60}\n")

    # Reload best weights before test evaluation
    model.load_state_dict(best_model_wts)
    test_loss, test_acc = evaluate(model, test_loader, criterion, split="Test")

    print(f"  ╔══════════════════════════════════════╗")
    print(f"  ║  TEST  Loss : {test_loss:.4f}                ║")
    print(f"  ║  TEST  Acc  : {test_acc:.4f}  ({test_acc*100:.2f}%)   ║")
    print(f"  ╚══════════════════════════════════════╝\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 13.  PER-CLASS METRICS  (confusion matrix style)
    # ──────────────────────────────────────────────────────────────────────────
    print("  Per-class breakdown on test set …")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Evaluating", leave=False):
            images = images.to(device)
            preds  = model(images).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    for idx, cls_name in enumerate(CLASS_NAMES):
        mask      = all_labels == idx
        cls_acc   = (all_preds[mask] == idx).mean() if mask.sum() > 0 else 0.0
        print(f"    {cls_name:<12s}: {cls_acc*100:.2f}%  ({mask.sum()} samples)")

    print(f"\n  Done. ✓\n")


if __name__ == "__main__":
    main()
