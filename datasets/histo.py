import os
import pandas as pd


def get_data(cfg: dict):
    class_names = cfg["class_names"]
    df          = pd.read_csv(cfg["csv_path"])

    image_paths = [
        os.path.join(cfg["data_dir"], row["id"] + ".tif")
        for _, row in df.iterrows()
    ]
    labels      = df["label"].tolist()
    num_classes = len(class_names)

    print(f"[histo] Obrázkov: {len(image_paths)} | Triedy: {class_names}")
    return image_paths, labels, class_names, num_classes