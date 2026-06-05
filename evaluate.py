# evaluate.py
# ─────────────────────────────────────────────────────
# Spustenie:
#   python evaluate.py --dataset xray --model resnet50
#   python evaluate.py --dataset skin --model densenet121
# ─────────────────────────────────────────────────────
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from tqdm import tqdm

from config import get_config, get_transforms
from models import get_model


# ==================================================
# ARGPARSE
# ==================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True,
                    help="xray | mri | skin | histo")
parser.add_argument("--model",   type=str, default="resnet50",
                    help="resnet50 | densenet121 | vgg19 | inceptionv3")
args = parser.parse_args()

cfg = get_config(args.dataset, args.model)
if args.model == "inceptionv3":
    cfg["img_size"] = 299


# ==================================================
# DEVICE
# ==================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")  # MPS má bug s VGG19 + img_size=96

print(f"Device: {device} | Dataset: {cfg['name']} | Model: {cfg['model_name']}")


# ==================================================
# DATASET
# ==================================================
class SimpleImageDataset(Dataset):
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
            h = cfg.get("img_size", 224)
            return torch.zeros(3, h, h), self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


_, val_transforms = get_transforms(cfg)


# ==================================================
# TEST LOADER
# ==================================================
def build_test_loader(cfg):
    name = cfg["name"]

    if name == "mri":
        from torchvision import datasets as tvd
        test_ds     = tvd.ImageFolder(cfg["test_dir"], transform=val_transforms)
        image_paths = [s[0] for s in test_ds.samples]
        labels      = test_ds.targets

    elif name == "xray":
        from torchvision import datasets as tvd
        base        = tvd.ImageFolder(cfg["test_dir"])
        image_paths = [s[0] for s in base.samples]
        labels      = base.targets

    elif name == "skin":
        import pandas as pd
        from sklearn.model_selection import train_test_split
        label_map = {cls: i for i, cls in enumerate(cfg["class_names"])}
        df        = pd.read_csv(cfg["csv_path"])
        df        = df.drop_duplicates(subset="lesion_id").reset_index(drop=True)
        df        = df[df["dx"].isin(cfg["class_names"])].reset_index(drop=True)

        def find_img(iid):
            for d in cfg["img_dirs"]:
                p = os.path.join(d, iid + ".jpg")
                if os.path.exists(p):
                    return p
            raise FileNotFoundError(iid)

        all_paths  = [find_img(r["image_id"]) for _, r in df.iterrows()]
        all_labels = [label_map[r["dx"]] for _, r in df.iterrows()]
        _, image_paths, _, labels = train_test_split(
            all_paths, all_labels,
            test_size=0.15, stratify=all_labels, random_state=cfg["seed"]
        )

    elif name == "histo":
        import pandas as pd
        from sklearn.model_selection import train_test_split
        df         = pd.read_csv(cfg["csv_path"])
        all_paths  = [os.path.join(cfg["data_dir"], r["id"] + ".tif")
                      for _, r in df.iterrows()]
        all_labels = df["label"].tolist()
        _, image_paths, _, labels = train_test_split(
            all_paths, all_labels,
            test_size=0.15, stratify=all_labels, random_state=cfg["seed"]
        )

    test_ds     = SimpleImageDataset(image_paths, labels, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=0)
    print(f"Test set: {len(test_ds)} obrázkov")
    return test_loader


test_loader = build_test_loader(cfg)


# ==================================================
# MODEL
# ==================================================
model_path = os.path.join(cfg["models_dir"], "best_model_cv.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model sa nenašiel: {model_path}\n"
        f"Spusti: python train.py --dataset {cfg['name']} --model {cfg['model_name']}"
    )

def adapt_head_to_checkpoint(model, model_name: str, state_dict: dict, num_classes: int):
    """Prispôsobí klasifikačnú hlavu modelu architektúre uloženej v checkpointe.

    Dôvod: niektoré staršie checkpointy (najmä skin modely) boli uložené
    s inou veľkosťou classifier/fc vrstiev. Bez tejto úpravy load_state_dict
    spadne na size mismatch.
    """

    if model_name in ["resnet50", "inceptionv3"]:
        if "fc.6.weight" in state_dict:
            if isinstance(model.fc, nn.Sequential):
                in_features = model.fc[0].in_features
            else:
                in_features = model.fc.in_features

            hidden1 = state_dict["fc.0.weight"].shape[0]
            hidden2 = state_dict["fc.3.weight"].shape[0]
            out_features = state_dict["fc.6.weight"].shape[0]

            model.fc = nn.Sequential(
                nn.Linear(in_features, hidden1),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden2, out_features),
            )

    elif model_name == "densenet121":
        if "classifier.0.weight" in state_dict and "classifier.3.weight" in state_dict:
            current_hidden = model.classifier[0].out_features
            checkpoint_hidden = state_dict["classifier.0.weight"].shape[0]

            if current_hidden != checkpoint_hidden:
                in_features = model.classifier[0].in_features
                out_features = state_dict["classifier.3.weight"].shape[0]
                model.classifier = nn.Sequential(
                    nn.Linear(in_features, checkpoint_hidden),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(checkpoint_hidden, out_features),
                )

    elif model_name == "vgg19":
        if "classifier.6.weight" in state_dict:
            hidden1 = state_dict["classifier.0.weight"].shape[0]
            hidden2 = state_dict["classifier.3.weight"].shape[0]
            out_features = state_dict["classifier.6.weight"].shape[0]

            model.classifier = nn.Sequential(
                nn.Linear(25088, hidden1),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden2, out_features),
            )

    return model


state_dict = torch.load(model_path, map_location=device)
model = get_model(cfg["model_name"], cfg["num_classes"])
model = adapt_head_to_checkpoint(model, cfg["model_name"], state_dict, cfg["num_classes"])
model = model.to(device)
model.load_state_dict(state_dict, strict=True)
model.eval()
print(f"Model načítaný: {model_path}")


# ==================================================
# PREDIKCIA
# ==================================================
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images  = images.to(device)
        outputs = model(images)
        preds   = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())


# ==================================================
# VÝSLEDKY
# ==================================================
class_names = cfg["class_names"]

print(f"\n{'='*60}")
print(f"  [{cfg['name']} | {cfg['model_name']}]")
print(f"{'='*60}")
print(classification_report(all_labels, all_preds, target_names=class_names))
print(f"Balanced Accuracy: {balanced_accuracy_score(all_labels, all_preds)*100:.2f}%")
print(f"{'='*60}\n")

os.makedirs(cfg["plots_dir"], exist_ok=True)
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(max(6, len(class_names)*1.5),
                                max(5, len(class_names)*1.2)))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predikované")
ax.set_ylabel("Skutočné")
ax.set_title(f"Confusion Matrix — {cfg['name']} | {cfg['model_name']}")
plt.tight_layout()
cm_path = os.path.join(cfg["plots_dir"], "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Confusion Matrix: {cm_path}")