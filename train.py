# train.py
# ─────────────────────────────────────────────────────
# Spustenie:
#   python train.py --dataset xray --model resnet50
#   python train.py --dataset skin --model densenet121
#   python train.py --dataset mri  --model inceptionv3
# ─────────────────────────────────────────────────────
import os
import sys
import csv
import random
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageFile
from sklearn.model_selection import StratifiedKFold, train_test_split
ImageFile.LOAD_TRUNCATED_IMAGES = True

from config   import get_config, get_transforms
from models   import get_model
from datasets import get_data


# ==================================================
# ARGPARSE
# ==================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True,
                    help="xray | mri | skin | histo")
parser.add_argument("--model",   type=str, default="resnet50",
                    help="resnet50 | densenet121 | vgg19 | inceptionv3")
parser.add_argument("--epochs",  type=int, default=None)
parser.add_argument("--splits",  type=int, default=None)
args = parser.parse_args()

cfg = get_config(args.dataset, args.model)
if args.epochs is not None:
    cfg["epochs"]   = args.epochs
if args.splits is not None:
    cfg["n_splits"] = args.splits

if args.model == "inceptionv3":
    cfg["img_size"] = 299


# ==================================================
# SEED + DEVICE
# ==================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(cfg["seed"])

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# ==================================================
# PRIEČINKY
# ==================================================
FOLDS_PLOTS_DIR   = os.path.join(cfg["plots_dir"], "folds")
SUMMARY_PLOTS_DIR = os.path.join(cfg["plots_dir"], "summary")

for d in [cfg["logs_dir"], cfg["models_dir"], FOLDS_PLOTS_DIR, SUMMARY_PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 70)
print(f"DATASET: {cfg['name']}  |  MODEL: {cfg['model_name']}")
print(f"DEVICE:  {device}  |  CLASSES: {cfg['class_names']}")
print(f"RESULTS: {cfg['results_dir']}")
print("=" * 70)


# ==================================================
# DATASET CLASS
# ==================================================
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception:
            return None
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


# ==================================================
# HELPERS
# ==================================================
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def create_weighted_sampler(train_labels):
    counts        = torch.bincount(torch.tensor(train_labels)).float()
    class_weights = 1.0 / counts
    sample_weights = [class_weights[l].item() for l in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, counts


def get_criterion(class_counts, device):
    weights   = 1.0 / class_counts
    weights   = weights / weights.sum()
    return nn.CrossEntropyLoss(weight=weights.to(device))


def run_eval(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images, targets = batch
            images, targets = images.to(device), targets.to(device)
            outputs         = model(images)
            loss            = criterion(outputs, targets)
            total_loss     += loss.item()
            total_correct  += (outputs.argmax(1) == targets).sum().item()
            total_samples  += targets.size(0)
    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / len(loader), 100.0 * total_correct / total_samples


# ==================================================
# DÁTA + FILTRÁCIA
# ==================================================
image_paths, labels, class_names, num_classes = get_data(cfg)

# Oddelenie held-out testu pre datasety bez samostatného testovacieho
# priečinka. Rovnaký seed a pomer ako v evaluate.py → bez úniku dát.
if "test_dir" not in cfg:
    image_paths, _, labels, _ = train_test_split(
        image_paths, labels,
        test_size=0.15, stratify=labels, random_state=cfg["seed"]
    )

train_transforms, val_transforms = get_transforms(cfg)

print("Overovanie neporušenosti súborov...")
good_paths, good_labels = [], []
for path, label in zip(image_paths, labels):
    try:
        with Image.open(path) as img:
            img.verify()
        good_paths.append(path)
        good_labels.append(label)
    except Exception:
        pass

image_paths = good_paths
labels      = good_labels
print(f"Súborov po filtrácii: {len(image_paths)}")


# ==================================================
# CSV LOGY
# ==================================================
epoch_csv   = os.path.join(cfg["logs_dir"], "epoch_log.csv")
summary_csv = os.path.join(cfg["logs_dir"], "fold_summary.csv")

with open(epoch_csv, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["dataset", "model", "fold", "epoch",
                             "train_loss", "train_acc", "val_loss", "val_acc"])
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["dataset", "model", "fold", "best_epoch", "best_val_acc"])


# ==================================================
# GRAFY
# ==================================================
def save_fold_plots(fold, history, out_dir):
    epochs = range(1, len(history["train_losses"]) + 1)
    for (t_key, v_key, title, ylabel, fname) in [
        ("train_losses", "val_losses", "Loss",     "Loss",          "loss"),
        ("train_accs",   "val_accs",   "Accuracy", "Accuracy (%)", "accuracy"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, history[t_key], label=f"Train", linewidth=2, color="steelblue")
        ax.plot(epochs, history[v_key], label=f"Val",   linewidth=2, color="tomato", linestyle="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — Fold {fold} [{cfg['model_name']}]")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fold{fold}_{fname}.png"), dpi=150)
        plt.close()


def save_summary_plots(all_history, all_best_accs, out_dir):
    fold_nums = list(range(1, len(all_best_accs) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(fold_nums, all_best_accs, color="steelblue", edgecolor="white")
    for bar, acc in zip(bars, all_best_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)
    mean_acc = np.mean(all_best_accs)
    ax.axhline(mean_acc, color="tomato", linestyle="--", label=f"Mean: {mean_acc:.2f}%")
    ax.set_xlabel("Fold"); ax.set_ylabel("Best Val Accuracy (%)")
    ax.set_title(f"Best Val Accuracy podľa foldov [{cfg['model_name']}]")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_acc_per_fold.png"), dpi=150)
    plt.close()

    min_len = min(len(h["train_losses"]) for h in all_history)
    mean_tl = np.mean([h["train_losses"][:min_len] for h in all_history], axis=0)
    mean_vl = np.mean([h["val_losses"][:min_len]   for h in all_history], axis=0)
    mean_ta = np.mean([h["train_accs"][:min_len]   for h in all_history], axis=0)
    mean_va = np.mean([h["val_accs"][:min_len]     for h in all_history], axis=0)
    ep = range(1, min_len + 1)

    for (y1, y2, title, ylabel, fname) in [
        (mean_tl, mean_vl, "Priemerný Loss",     "Loss",          "mean_loss"),
        (mean_ta, mean_va, "Priemerná Accuracy", "Accuracy (%)", "mean_accuracy"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ep, y1, label="Train", linewidth=2, color="steelblue")
        ax.plot(ep, y2, label="Val",   linewidth=2, color="tomato", linestyle="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(f"{title} [{cfg['model_name']}]")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{fname}.png"), dpi=150)
        plt.close()


# ==================================================
# CROSS-VALIDATION
# ==================================================
skf = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])

all_best_accs = []
all_history   = []
global_best   = 0.0
global_fold   = 0

for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels), start=1):
    print(f"\n{'='*70}")
    print(f"[{cfg['name']}|{cfg['model_name']}] FOLD {fold}/{cfg['n_splits']}")
    print(f"{'='*70}")

    train_paths  = [image_paths[i] for i in train_idx]
    val_paths    = [image_paths[i] for i in val_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels   = [labels[i] for i in val_idx]

    sampler, class_counts = create_weighted_sampler(train_labels)

    train_ds = CustomImageDataset(train_paths, train_labels, train_transforms)
    val_ds   = CustomImageDataset(val_paths,   val_labels,   val_transforms)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Class counts: { {class_names[i]: int(class_counts[i]) for i in range(num_classes)} }")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              sampler=sampler, num_workers=0,
                              collate_fn=collate_skip_none)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=0,
                              collate_fn=collate_skip_none)

    model     = get_model(cfg["model_name"], num_classes).to(device)
    criterion = get_criterion(class_counts, device)
    if cfg["model_name"] == "resnet50":
        optimizer = Adam([
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(),     "lr": 1e-3},
        ])
    else:
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"])

    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=3, min_lr=1e-6, verbose=True)

    best_val_acc      = 0.0
    best_epoch        = 0
    epochs_no_improve = 0
    fold_history = {"train_losses": [], "val_losses": [],
                    "train_accs":   [], "val_accs":   []}

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        # ResNet50: zmrazené BN vrstvy držíme v evaluačnom režime aj počas tréningu.
        if cfg["model_name"] == "resnet50":
            for m in (model.conv1, model.bn1, model.layer1, model.layer2, model.layer3):
                m.eval()
        run_loss, run_correct, run_total = 0.0, 0, 0

        for batch in train_loader:
            if batch is None:
                continue
            images, targets = batch
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            run_loss    += loss.item()
            run_correct += (outputs.argmax(1) == targets).sum().item()
            run_total   += targets.size(0)

        if run_total == 0:
            continue

        train_loss = run_loss / len(train_loader)
        train_acc  = 100.0 * run_correct / run_total
        val_loss, val_acc = run_eval(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        fold_history["train_losses"].append(train_loss)
        fold_history["val_losses"].append(val_loss)
        fold_history["train_accs"].append(train_acc)
        fold_history["val_accs"].append(val_acc)

        print(f"[{cfg['model_name']}] Fold {fold}/{cfg['n_splits']} | "
              f"Epoch {epoch:02d}/{cfg['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        with open(epoch_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([cfg["name"], cfg["model_name"], fold, epoch,
                                    round(train_loss,6), round(train_acc,4),
                                    round(val_loss,6),   round(val_acc,4)])

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            best_epoch        = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(cfg["models_dir"], f"best_fold{fold}.pth"))
            if val_acc > global_best:
                global_best = val_acc
                global_fold = fold
                torch.save(model.state_dict(),
                           os.path.join(cfg["models_dir"], "best_model_cv.pth"))
            print(f"  ✓ Uložený best model fold {fold}")
        else:
            epochs_no_improve += 1
            print(f"  ⚠ Žiadne zlepšenie: {epochs_no_improve}/{cfg['patience']}")
            if epochs_no_improve >= cfg["patience"]:
                print(f"  🛑 Early stopping na epoch {epoch}")
                break

    save_fold_plots(fold, fold_history, FOLDS_PLOTS_DIR)
    print(f"  📊 Grafy uložené → plots/folds/fold{fold}_*.png")

    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([cfg["name"], cfg["model_name"],
                                 fold, best_epoch, round(best_val_acc, 4)])

    all_best_accs.append(best_val_acc)
    all_history.append(fold_history)
    print(f"Best val acc fold {fold}: {best_val_acc:.2f}% (epoch {best_epoch})")


# ==================================================
# SUMMARY
# ==================================================
save_summary_plots(all_history, all_best_accs, SUMMARY_PLOTS_DIR)
mean_acc = np.mean(all_best_accs)

print("\n" + "=" * 70)
print(f"[{cfg['name']}|{cfg['model_name']}] HOTOVO")
print(f"Mean val accuracy: {mean_acc:.2f}%")
print(f"Best fold: {global_fold} | Best acc: {global_best:.2f}%")
print(f"Model: {os.path.join(cfg['models_dir'], 'best_model_cv.pth')}")
print("=" * 70)