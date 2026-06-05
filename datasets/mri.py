from torchvision import datasets


def get_data(cfg: dict):
    base = datasets.ImageFolder(cfg["data_dir"])
    image_paths = [s[0] for s in base.samples]
    labels      = base.targets
    class_names = base.classes
    num_classes = len(class_names)
    print(f"[mri] Obrázkov: {len(image_paths)} | Triedy: {class_names}")
    return image_paths, labels, class_names, num_classes