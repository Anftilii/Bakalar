# Convolutional Neural Networks for Medical Image Classification

Bachelor's thesis — Technical University of Košice, Faculty of Electrical Engineering and Informatics (TUKE FEI), 2026.

Experimental comparison of four CNN architectures (**VGG19**, **ResNet50**, **DenseNet121**, **InceptionV3**) on four medical image classification tasks using transfer learning and 10-fold cross-validation.

## Datasets

| Dataset | Task | Classes |
|---|---|---|
| [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | Binary | NORMAL, PNEUMONIA |
| [Brain Tumor MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | Multi-class | glioma, meningioma, notumor, pituitary |
| [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) | Multi-class | akiec, bcc, bkl, mel, nv |
| [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection) | Binary | no\_cancer, cancer |

Datasets are not included in this repository. Download them from Kaggle and arrange according to the structure described in `config.py`.

## Requirements

- Python 3.10+
- PyTorch 2.0+ and torchvision 0.15+ (install from [pytorch.org](https://pytorch.org/get-started/locally/) matching your CUDA version)

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` and set the paths to your data and results directories:

```python
BASE_DATA    = "/path/to/data"
BASE_RESULTS = "/path/to/results"
```

For the HAM10000 dataset, also update `data_dir`, `csv_path` and `img_dirs` in the `skin` config section.

## Usage

**Train a model:**
```bash
python train.py --dataset xray --model inceptionv3
python train.py --dataset mri  --model densenet121
python train.py --dataset skin --model resnet50
python train.py --dataset histo --model resnet50
```

Optional arguments: `--epochs N`, `--splits K`

**Evaluate a trained model:**
```bash
python evaluate.py --dataset xray --model inceptionv3
```

**Run all combinations:**
```bash
bash run_all.sh            # all models × all datasets
bash run_all.sh resnet50   # one model, all datasets
bash run_test.sh           # quick check (2 folds, 2 epochs)
```


## Project Structure

```
├── config.py          dataset configs, hyperparameters, paths
├── train.py           training with stratified k-fold CV
├── evaluate.py        final evaluation on test set
├── run_all.sh         batch training script
├── run_test.sh        quick sanity check
├── datasets/          data loading modules
├── models/            CNN architecture definitions
└── requirements.txt
```