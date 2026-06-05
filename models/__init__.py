# models/__init__.py
# ─────────────────────────────────────────────────────
# Dispečer modelov — vracia požadovaný model podľa mena.
# Použitie:
#   from models import get_model
#   model = get_model("resnet50", num_classes=5)
# ─────────────────────────────────────────────────────
from .resnet50    import build_model as _resnet50
from .densenet121 import build_model as _densenet121
from .vgg19       import build_model as _vgg19
from .inceptionv3 import build_model as _inceptionv3


def get_model(model_name: str, num_classes: int):
    models = {
        "resnet50":    _resnet50,
        "densenet121": _densenet121,
        "vgg19":       _vgg19,
        "inceptionv3": _inceptionv3,
    }
    assert model_name in models, (
        f"Neznámy model '{model_name}'. "
        f"Dostupné: {list(models.keys())}"
    )
    return models[model_name](num_classes)