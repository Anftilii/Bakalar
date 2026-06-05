# models/inceptionv3.py
# ─────────────────────────────────────────────────────
# DÔLEŽITÉ: InceptionV3 očakáva img_size=299, nie 224!
# V config.py treba pre datasety s inception nastaviť img_size=299
# alebo ho predať samostatne cez --imgsize 299.
# ─────────────────────────────────────────────────────
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int) -> nn.Module:
    model = models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT,
        aux_logits=True,   # pomocný výstup — vypneme nižšie
    )
    model.aux_logits = False  # odstránime auxiliary classifier

    # Zmrazíme všetko okrem Mixed_7 (posledný inception blok) a fc
    for name, param in model.named_parameters():
        param.requires_grad = "Mixed_7" in name or "fc" in name

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return model