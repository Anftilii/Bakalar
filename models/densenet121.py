# models/densenet121.py
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int) -> nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    # Zmrazíme všetko okrem denseblock4 (posledný blok) a classifier
    for name, param in model.named_parameters():
        param.requires_grad = "denseblock4" in name or "classifier" in name

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return model