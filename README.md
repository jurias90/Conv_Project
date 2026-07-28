# OCT Retinal Disease Classifier

Deep learning model for classifying OCT retinal scans into 5 categories using
transfer learning on VGG16 pretrained on ImageNet. Includes out-of-distribution
detection — the model can recognize when an image is NOT an OCT scan.

## Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| Kermany et al. 2018 (paper) | 96.6% | Inception-v3, full dataset |
| Mooney Kaggle notebook | 91.7% | VGG16, 752 samples, frozen |
| ResNet50 (this project) | 98.76% | ResNet50, full dataset, layer4 unfrozen |
| VGG16 4-class (this project) | **99.48%** | VGG16, full dataset, block5 unfrozen |
| VGG16 5-class (this project) | **96.90%** | VGG16 + NOT_OCT rejection, 100% rejection accuracy |

## Experiment Log

| Run | Change | Test Accuracy |
|-----|--------|---------------|
| 1 | Mooney notebook baseline (Kaggle, 752 samples) | 91.7% |
| 2 | VGG16 frozen, full dataset, lr=0.001 | 83.1% |
| 3 | VGG16 frozen, full dataset, lr=0.003 | 74.3% |
| 4 | VGG16 block5 unfrozen, lr=0.001/0.0001 | **99.48%** |
| 5 | ResNet50 layer4 unfrozen, lr=0.001/0.0001 | 98.76% |
| 6 | VGG16 5-class (+ NOT_OCT rejection) | **96.90%** |

## Per-Class Accuracy (5-class model)

| Class | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| CNV | 3582 | 3634 | 98.57% |
| DME | 1122 | 1172 | 95.73% |
| DRUSEN | 736 | 899 | 81.87% |
| NORMAL | 2577 | 2621 | 98.32% |
| NOT_OCT | 1640 | 1640 | **100.00%** |
| **Total** | **9657** | **9966** | **96.90%** |

---

## Project Structure

```
Conv_Project/
├── oct_train.py           ← 4-class VGG16 training pipeline
├── resnet50_train.py      ← 4-class ResNet50 training pipeline
├── oct_5class_train.py    ← 5-class VGG16 with NOT_OCT rejection
├── inference.py           ← diagnose a single image
├── setup_data.py          ← builds combined_data folder automatically
├── oct_model.pth          ← saved VGG16 4-class weights (~500MB)
├── resnet50_model.pth     ← saved ResNet50 weights (~100MB)
├── oct_5class_model.pth   ← saved 5-class weights (~500MB)
├── requirements.txt       ← python dependencies
├── README.md              ← this file
├── data/
│   └── OCT2017/
│       ├── train/
│       │   ├── CNV/
│       │   ├── DME/
│       │   ├── DRUSEN/
│       │   └── NORMAL/
│       ├── test/
│       └── val/
├── not_oct_data/          ← raw downloaded non-OCT datasets
│   ├── chest_xray/        ← NIH chest X-rays
│   ├── cifar10/           ← everyday photos
│   ├── fundus/            ← APTOS retinal fundus photos
│   └── skin/              ← skin lesion images
└── combined_data/         ← merged dataset for 5-class training
    ├── CNV/
    ├── DME/
    ├── DRUSEN/
    ├── NORMAL/
    └── NOT_OCT/           ← sampled from all 4 not_oct_data sources
```

---

## Setup

### Step 1 — Clone or copy the project folder

Files needed at minimum:
- `oct_train.py`
- `oct_5class_train.py`
- `inference.py`
- `setup_data.py`
- `requirements.txt`
- `oct_model.pth` (for 4-class inference)
- `oct_5class_model.pth` (for 5-class inference with rejection)

---

### Step 2 — Create a virtual environment

**On Linux (CachyOS) or Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

**In PyCharm:**
- `File → New Project` → let PyCharm create the venv automatically
- Open the Terminal tab — it will already be inside the venv

---

### Step 3 — Install PyTorch

PyTorch must be installed separately before other packages.

**MacBook M5 Pro (Apple Silicon):**
```bash
pip install torch torchvision
```

Verify:
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```
Should print: `MPS available: True`

---

**Linux with NVIDIA RTX 5080 (CUDA 13.2):**
```bash
pip install torch==2.14.0.dev20260706 \
  --index-url https://download.pytorch.org/whl/nightly/cu132
pip install torchvision \
  --index-url https://download.pytorch.org/whl/nightly/cu132
```

Verify:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Should print: `True` and `NVIDIA GeForce RTX 5080`

---

**Other NVIDIA GPU — check CUDA version first:**
```bash
nvidia-smi
```

| CUDA Version | Install Command |
|-------------|-----------------|
| 12.4 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| 12.1 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| 11.8 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| CPU only | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |

---

### Step 4 — Install remaining packages
```bash
pip install -r requirements.txt
```

---

### Step 5 — Download the OCT dataset

**Option A: Kaggle CLI**
```bash
mkdir -p ~/.kaggle
# Place kaggle.json in ~/.kaggle/ (get from kaggle.com → Settings → API)
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d paultimothymooney/kermany2018
unzip kermany2018.zip -d data/
```

**Option B: Manual download (no account needed)**

Go to: https://data.mendeley.com/datasets/rscbjbr9sj/3
Click Download, unzip into `Conv_Project/data/`

---

### Step 6 — Download NOT_OCT datasets (for 5-class model only)

Requires a free Kaggle account. Get your API token at:
kaggle.com → profile icon → Settings → API → Create New Token

Storage required: ~63GB total free space

```bash
mkdir -p not_oct_data

# CIFAR-10 everyday photos (~170MB)
kaggle datasets download -d gazu468/cifar10-classification-image
unzip cifar10-classification-image.zip -d not_oct_data/cifar10

# NIH Chest X-rays (~45GB — largest download)
kaggle datasets download -d nih-chest-xrays/data
unzip data.zip -d not_oct_data/chest_xray

# APTOS fundus eye photos (~9GB)
kaggle datasets download -d mariaherrerot/aptos2019
unzip aptos2019.zip -d not_oct_data/fundus

# Skin lesion images (~3GB)
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d not_oct_data/skin
```

---

### Step 7 — Build combined_data folder

Once all datasets are downloaded, run the setup script:
```bash
python setup_data.py
```

This automatically:
- Creates the combined_data folder structure
- Copies all OCT training images
- Randomly samples 5,000 images from each NOT_OCT source
- Prints a summary of image counts when done

Expected output:
```
CNV:     37,205
DME:     11,348
DRUSEN:   8,616
NORMAL:  26,315
NOT_OCT: ~18,000
Total:   ~101,000 images
```

---

## Running Inference (diagnose a single image)

```bash
python inference.py path/to/scan.jpg
```

Example with a known OCT scan:
```bash
python inference.py data/OCT2017/test/CNV/CNV-1016042-1.jpeg
```

Example with a non-OCT image (should be rejected):
```bash
python inference.py path/to/random_photo.jpg
```

Expected output for a valid OCT scan:
```
Running on: cuda  (or mps on MacBook)
Model loaded successfully

==================================================
Image:      CNV-1016042-1.jpeg
Diagnosis:  CNV
Confidence: 99.12%
Meaning:    Choroidal Neovascularization — abnormal blood vessel growth

All class probabilities:
  CNV      99.12% ██████████████████████████████
  DME       0.51%
  DRUSEN    0.24%
  NORMAL    0.13%
  NOT_OCT   0.00%
==================================================
```

Expected output for a non-OCT image:
```
==================================================
Image:      random_photo.jpg
Diagnosis:  NOT_OCT
Confidence: 94.3%
Meaning:    Image does not appear to be an OCT retinal scan

All class probabilities:
  CNV       1.2%
  DME       0.8%
  DRUSEN    2.1%
  NORMAL    1.6%
  NOT_OCT  94.3% ████████████████████████████
==================================================
```

---

## Running Training

**4-class model (OCT only):**
```bash
python oct_train.py
```
~45 minutes on RTX 5080. Saves to `oct_model.pth`.

**5-class model (OCT + NOT_OCT rejection):**
```bash
python oct_5class_train.py
```
~55 minutes on RTX 5080. Saves to `oct_5class_model.pth`.

**ResNet50 comparison:**
```bash
python resnet50_train.py
```
~25 minutes on RTX 5080. Saves to `resnet50_model.pth`.

---

## MacBook M5 Pro Notes

- Uses Apple's MPS backend instead of CUDA — automatically detected
- Training is slower (~6 min/epoch vs ~4 min/epoch on RTX 5080)
- All training scripts handle MPS automatically via device detection:
```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```
- Set `pin_memory=False` in DataLoaders when running on Mac

---

## Model Details

### VGG16 4-class (oct_train.py)

| Parameter | Value |
|-----------|-------|
| Architecture | VGG16 |
| Pretrained on | ImageNet (IMAGENET1K_V1) |
| Frozen layers | features.0 through features.23 (blocks 1-4) |
| Unfrozen layers | features.24-28 (block5) |
| New head | nn.Linear(4096, 4) |
| Trainable params | 7,095,812 / 134,276,932 |
| Optimizer | Adam (classifier lr=0.001, features lr=0.0001) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Epochs | 10 |
| Batch size | 32 |
| Test accuracy | 99.48% (963/968) |

### VGG16 5-class (oct_5class_train.py)

| Parameter | Value |
|-----------|-------|
| Architecture | VGG16 |
| Classes | CNV, DME, DRUSEN, NORMAL, NOT_OCT |
| New head | nn.Linear(4096, 5) |
| Trainable params | 7,099,909 / 134,281,029 |
| NOT_OCT sources | CIFAR-10, NIH Chest X-rays, APTOS fundus, skin lesions |
| NOT_OCT images | 16,176 |
| Optimizer | Adam (classifier lr=0.001, features lr=0.0001) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Epochs | 10 |
| Batch size | 32 |
| Test accuracy | 96.90% (9657/9966) |
| NOT_OCT rejection | 100.00% (1640/1640) |

### ResNet50 (resnet50_train.py)

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet50 |
| Unfrozen layers | layer4 |
| New head | nn.Linear(2048, 4) |
| Trainable params | 14,985,226 / 23,528,522 |
| Test accuracy | 98.76% (956/968) |

---

## Classes

### 4-class model
| Label | Full Name | Description |
|-------|-----------|-------------|
| CNV | Choroidal Neovascularization | Abnormal blood vessel growth beneath retina |
| DME | Diabetic Macular Edema | Fluid accumulation in the macula |
| DRUSEN | Drusen | Deposits under retina, early macular degeneration sign |
| NORMAL | Normal | Healthy retina, no pathology detected |

### 5-class model (adds)
| Label | Description |
|-------|-------------|
| NOT_OCT | Image is not an OCT retinal scan — rejected |

---

## Key Findings

**1. VGG16 outperforms ResNet50 on this dataset:**
```
VGG16    (7.1M trainable params)  → 99.48% test accuracy
ResNet50 (14.9M trainable params) → 98.76% test accuracy
```
ResNet50's layer4 contains 63% of its total parameters — too much capacity
for 66,787 training images, causing overfitting. Val loss climbed from
0.17 to 0.29 over 10 epochs while VGG16's stayed stable.
Dataset size favored the smaller trainable parameter count.

**2. Adding NOT_OCT class improves OCT classification accuracy:**
```
4-class val accuracy at epoch 3: 0.9545
5-class val accuracy at epoch 3: 0.9672
```
Forcing the model to distinguish OCT from non-OCT images pushes it to
learn sharper, more discriminative retinal features — improving accuracy
on the OCT classes as a side effect.

**3. Perfect NOT_OCT rejection (100%):**
Every non-OCT image in the test set was correctly identified as not being
a retinal scan. Sources included chest X-rays, everyday photos, fundus
photos, and skin lesion images — covering easy and hard rejection cases.

**4. DRUSEN is the hardest class (81.87%):**
DRUSEN deposits are visually subtle and share features with both NORMAL
retinas and early CNV. This aligns with clinical reality — DRUSEN is
the earliest and most ambiguous sign of macular degeneration, making it
the most challenging condition for both human clinicians and AI models.

---

## References

- Kermany et al. (2018). Identifying Medical Diagnoses and Treatable Diseases
  by Image-Based Deep Learning. Cell, 172(5), 1122-1131.
  https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

- Kaggle notebook (Mooney):
  https://www.kaggle.com/code/paultimothymooney/detect-retina-damage-from-oct-images

- Published paper using same approach (VGG16 + transfer learning on OCT):
  https://link.springer.com/article/10.1007/s42452-025-06565-6
  Their result: 95.19% — this project achieved 99.48% (4-class)

- NIH Chest X-ray dataset:
  https://www.kaggle.com/datasets/nih-chest-xrays/data

- APTOS 2019 Blindness Detection (fundus photos):
  https://www.kaggle.com/datasets/mariaherrerot/aptos2019

- HAM10000 Skin Lesion dataset:
  https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000