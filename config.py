# config.py
# ─────────────────────────────────────────────────────
# Konfigurácia pre RunPod.
# Dáta sa očakávajú v /workspace/data/
# Výsledky sa ukladajú v /workspace/results/
# ─────────────────────────────────────────────────────
import os
from torchvision import transforms

# ─────────────────────────────────────────────────────
# ZÁKLADNÉ CESTY (RunPod)
# ─────────────────────────────────────────────────────
BASE_DATA    = "/Users/toliaporplycia/Desktop/bakalarka/practik/data"
BASE_RESULTS = "/Users/toliaporplycia/Desktop/bakalarka/practik/results"

AVAILABLE_MODELS = ["resnet50", "densenet121", "vgg19", "inceptionv3"]

# ─────────────────────────────────────────────────────
# KONFIGURÁCIE DATASETOV
# ─────────────────────────────────────────────────────
DATASET_CONFIGS = {

    "xray": {
        "data_dir":     os.path.join(BASE_DATA, "xray", "train"),
        "test_dir":     os.path.join(BASE_DATA, "xray", "test"),
        "num_classes":  2,
        "class_names":  ["NORMAL", "PNEUMONIA"],
        "img_size":     224,
        "batch_size":   64,       # na GPU môže byť viac
        "augmentation": "standard",
        "loader_type":  "image_folder",
        "patience":     5,
    },

    "mri": {
        "data_dir":     os.path.join(BASE_DATA, "mri", "Training"),
        "test_dir":     os.path.join(BASE_DATA, "mri", "Testing"),
        "num_classes":  4,
        "class_names":  ["glioma", "meningioma", "notumor", "pituitary"],
        "img_size":     224,
        "batch_size":   64,
        "augmentation": "standard",
        "loader_type":  "image_folder",
        "patience":     5,
    },

    "skin": {
        "data_dir":     "/Users/toliaporplycia/Desktop/bakalarka/practik/data/foto",
        "num_classes":  5,
        "class_names":  ["akiec", "bcc", "bkl", "mel", "nv"],
        "img_size":     224,
        "batch_size":   32,
        "augmentation": "skin",
        "loader_type":  "csv",
        "csv_path":     "/Users/toliaporplycia/Desktop/bakalarka/practik/data/foto/HAM10000_metadata.csv",
        "img_dirs": [
            "/Users/toliaporplycia/Desktop/bakalarka/practik/data/foto/HAM10000_images_part_1",
            "/Users/toliaporplycia/Desktop/bakalarka/practik/data/foto/HAM10000_images_part_2",
        ],
        "patience":     10,
        "lr":           0.0003,
    },

    "histo": {
        "data_dir":     os.path.join(BASE_DATA, "histo", "train"),
        "num_classes":  2,
        "class_names":  ["no_cancer", "cancer"],
        "img_size":     96,
        "batch_size":   128,      # histo malé obrázky — môže byť veľký batch
        "augmentation": "histo",
        "loader_type":  "csv",
        "csv_path":     os.path.join(BASE_DATA, "histo", "train_labels.csv"),
        "patience":     7,
    },
}

# ─────────────────────────────────────────────────────
# HYPERPARAMETRE
# ─────────────────────────────────────────────────────
EPOCHS   = 50
LR       = 0.001
SEED     = 42
N_SPLITS = 10

# ─────────────────────────────────────────────────────
# TRANSFORMÁCIE
# ─────────────────────────────────────────────────────
_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std= [0.229, 0.224, 0.225],
)

def get_transforms(cfg: dict):
    size = cfg.get("img_size", 224)
    aug  = cfg.get("augmentation", "standard")

    val_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        _normalize,
    ])

    if aug == "standard":
        train_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            _normalize,
        ])
    elif aug == "skin":
        train_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            _normalize,
        ])
    elif aug == "histo":
        train_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            _normalize,
        ])
    else:
        raise ValueError(f"Neznámy typ augmentácie: '{aug}'")

    return train_transforms, val_transforms


# ─────────────────────────────────────────────────────
# get_config()
# ─────────────────────────────────────────────────────
def get_config(dataset_name: str, model_name: str = "resnet50") -> dict:
    assert dataset_name in DATASET_CONFIGS, (
        f"Neznámy dataset '{dataset_name}'. "
        f"Dostupné: {list(DATASET_CONFIGS.keys())}"
    )
    assert model_name in AVAILABLE_MODELS, (
        f"Neznámy model '{model_name}'. Dostupné: {AVAILABLE_MODELS}"
    )

    cfg = DATASET_CONFIGS[dataset_name].copy()
    cfg["name"]       = dataset_name
    cfg["model_name"] = model_name
    cfg["epochs"]     = EPOCHS
    cfg["seed"]       = SEED
    cfg["n_splits"]   = N_SPLITS
    cfg["lr"]         = cfg.get("lr", LR)  # berie sa z datasetu ak existuje

    results_dir        = os.path.join(BASE_RESULTS, f"{dataset_name}_{model_name}")
    cfg["results_dir"] = results_dir
    cfg["logs_dir"]    = os.path.join(results_dir, "logs")
    cfg["models_dir"]  = os.path.join(results_dir, "models")
    cfg["plots_dir"]   = os.path.join(results_dir, "plots")

    return cfg