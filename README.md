# Calibrated Lightweight Convolutional Ensemble for Multi-Class Skin Lesion Classification

Reproducible training and evaluation code for a calibrated six-backbone
convolutional neural network (CNN) ensemble that classifies seven dermoscopic
diagnoses, with **lesion-identifier-aware partitioning** on HAM10000 and a
**zero-shot external evaluation** on the ISIC 2019 release.

This repository accompanies the manuscript
*"A Calibrated Lightweight Convolutional Ensemble with Lesion-Identifier-Aware
Partitioning for Multi-Class Skin Lesion Classification: Development and
External Cohort Evaluation."* It contains only the data, training, calibration,
and evaluation pipeline; figure and manuscript-preparation scripts are not
included.

## What it does

- Downloads HAM10000 and ISIC 2019 images from the public ISIC Archive
  (`src/fetch_images.py`).
- Builds **lesion-grouped**, stratified three-fold cross-validation so no lesion
  appears in both training and test folds.
- Fine-tunes six ImageNet-pretrained backbones under one identical recipe with
  class-balanced focal loss: EfficientNet-B0, MobileNetV2, MobileNetV3-Small,
  ShuffleNetV2 ×1.0, ResNet-50, and DenseNet-121.
- Fits a per-model **temperature scaling** on validation data and reports
  Expected Calibration Error before and after.
- Forms four ensembles (soft and validation-weighted voting, three or six
  members) and evaluates discrimination, calibration, per-class metrics,
  compute profile, and image-level bootstrap uncertainty.
- Runs a **zero-shot external evaluation** on the non-overlapping ISIC 2019
  subset (`src/external_eval.py`).

## Repository layout

```
.
├── notebooks/
│   └── skin_lesion_ensemble.ipynb   # end-to-end pipeline (runnable)
├── src/
│   ├── fetch_images.py              # download HAM10000 + ISIC 2019 thumbnails
│   ├── train_eval.py                # training + internal CV evaluation
│   ├── external_eval.py             # zero-shot ISIC 2019 evaluation
│   └── export_tables.py             # results CSV -> LaTeX tables
├── configs/
│   └── default.yaml                 # hyper-parameters and paths
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick start

```bash
python -m venv .venv && source .venv/Scripts/activate   # or .venv/bin/activate
pip install -r requirements.txt

# 1. fetch images (writes data/ham10000/images and data/isic2019/images)
python src/fetch_images.py ham isic

# 2. train + internal cross-validation (set CF_SMOKE=1 for a 1-epoch smoke test)
python src/train_eval.py

# 3. external zero-shot evaluation on ISIC 2019
python src/external_eval.py
```

The notebook `notebooks/skin_lesion_ensemble.ipynb` runs the same pipeline
cell by cell.

## Data

- **HAM10000** — Harvard Dataverse, DOI
  [10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T).
- **ISIC 2019** — International Skin Imaging Collaboration Challenge data.

Images are retrieved from the public ISIC Archive. The scripts store only
derived image files and prediction arrays; no patient-identifiable data are
distributed here.

## Reproducibility

- Seed `42`; three stratified lesion-grouped folds.
- Per-fold out-of-fold probabilities are cached under `results/` so an
  interrupted run resumes without recomputation.
- PyTorch 2.6 / CUDA 12.4; a single 12 GB GPU is sufficient.

## Environment note

Training was run on a single NVIDIA RTX 3060 (12 GB). Mixed-precision training
and a batch size of 32 fit within that budget.

## Citation

If you use this code, please cite the accompanying manuscript (details to be
added on acceptance).

## License

Released under the MIT License (see `LICENSE`). The HAM10000 and ISIC datasets
retain their own licenses and terms of use.
