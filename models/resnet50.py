# models/resnet50.py
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Zmrazenie BN štatistík v predtrénovanej časti (conv1, bn1, layer1-3)
    for module in [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3]:
        if hasattr(module, 'eval'):
            module.eval()

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return model