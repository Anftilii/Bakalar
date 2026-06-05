import os
import pandas as pd

def get_data(cfg: dict):
    class_names = cfg["class_names"]
    label_map   = {cls: i for i, cls in enumerate(class_names)}

    df = pd.read_csv(cfg["csv_path"])
    df = df.drop_duplicates(subset="lesion_id").reset_index(drop=True)
    df = df[df["dx"].isin(class_names)].reset_index(drop=True)

    def find_image(image_id):
        for d in cfg["img_dirs"]:
            path = os.path.join(d, image_id + ".jpg")
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Obrázok {image_id}.jpg sa nenašiel")

    image_paths = [find_image(row["image_id"]) for _, row in df.iterrows()]
    labels      = [label_map[row["dx"]] for _, row in df.iterrows()]
    num_classes = len(class_names)

    print(f"[skin] Obrázkov: {len(image_paths)} | Triedy: {class_names}")
    return image_paths, labels, class_names, num_classes