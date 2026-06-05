# Konvolučné neurónové siete pre klasifikáciu medicínskych obrazov

Bakalárska práca — Technická univerzita v Košiciach, Fakulta elektrotechniky a informatiky (TUKE FEI), 2026.

Experimentálne porovnanie architektúr **VGG19**, **ResNet50**, **DenseNet121** a **InceptionV3** na štyroch úlohách klasifikácie medicínskych obrazov pomocou transferového učenia a 10-násobnej krížovej validácie.

## Datasety

| Dataset | Úloha | Triedy |
|---|---|---|
| [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | Binárna | NORMAL, PNEUMONIA |
| [Brain Tumor MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | Viactriedna | glioma, meningioma, notumor, pituitary |
| [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) | Viactriedna | akiec, bcc, bkl, mel, nv |
| [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection) | Binárna | no\_cancer, cancer |

Datasety nie sú súčasťou repozitára — stiahni ich z Kaggle a nastav cesty v `config.py`.

## Inštalácia

Python 3.10+, PyTorch 2.0+ (inštalácia podľa [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
pip install -r requirements.txt
```

## Konfigurácia

V súbore `config.py` nastav cesty k dátam a výsledkom:

```python
BASE_DATA    = "/cesta/k/datam"
BASE_RESULTS = "/cesta/k/vysledkom"
```

## Spustenie

```bash
# Tréning
python train.py --dataset xray --model inceptionv3
python train.py --dataset mri  --model densenet121 --epochs 30 --splits 5

# Vyhodnotenie
python evaluate.py --dataset xray --model inceptionv3

# Hromadné spustenie
bash run_all.sh
bash run_test.sh   # rychla kontrola (2 foldy, 2 epochy)
```

## Výsledky

| Dataset | Najlepší model | Testovacia presnosť | Vyvážená presnosť |
|---|---|---|---|
| Chest X-Ray | InceptionV3 | 87,34 % | 83,38 % |
| Brain Tumor MRI | DenseNet121 | 95,00 % | 95,00 % |
| HAM10000 | ResNet50 | 97,00 % | 96,55 % |
| Histopathologic Cancer | ResNet50 | 97,00 % | 97,11 % |
