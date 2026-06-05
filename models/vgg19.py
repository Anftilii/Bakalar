# models/vgg19.py
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int) -> nn.Module:
    model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

    # Zmrazíme features, rozmrazíme iba classifier
    for param in model.features.parameters():
        param.requires_grad = False

    # VGG19 má veľký classifier — nahradíme ho celý
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return model