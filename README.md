# OCT Retinal Disease Classifier

Deep learning model for classifying OCT retinal scans into 4 categories using
transfer learning on VGG16 pretrained on ImageNet.

## Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| Kermany et al. 2018 (paper) | 96.6% | Inception-v3, full dataset |
| Mooney Kaggle notebook | 91.7% | VGG16, 752 samples, frozen |
| This project | **99.48%** | VGG16, full dataset, block5 unfrozen |

## Project Structure

```
Conv_Project/
├── oct_train.py       ← full training pipeline
├── inference.py       ← diagnose a single image
├── oct_model.pth      ← saved weights (~500MB)
├── requirements.txt   ← python dependencies
├── README.md          ← this file
└── data/
    └── OCT2017/
        ├── train/
        │   ├── CNV/
        │   ├── DME/
        │   ├── DRUSEN/
        │   └── NORMAL/
        ├── test/
        └── val/
```

---

## Setup

### Step 1 — Clone or copy the project folder

Make sure you have these files at minimum:
- `oct_train.py`
- `inference.py`
- `requirements.txt`
- `oct_model.pth` (needed for inference, not for training from scratch)

---

### Step 2 — Create a virtual environment

Open a terminal in the project folder, then:

**On Linux (CachyOS) or Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt.

**In PyCharm (either machine):**
- `File → New Project` → let PyCharm create the venv automatically
- Open the built-in Terminal tab at the bottom — it will already be inside the venv

---

### Step 3 — Install PyTorch

PyTorch must be installed separately before the other packages because
the install URL depends on your hardware.

**MacBook M5 Pro (Apple Silicon):**
```bash
pip install torch torchvision
```
That's it — Apple Silicon support is built into the standard PyTorch release.

Verify it worked:
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

Verify it worked:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Should print: `True` and `NVIDIA GeForce RTX 5080`

---

**Other NVIDIA GPU — check your CUDA version first:**
```bash
nvidia-smi
```
Look for `CUDA Version: X.X` in the top right, then use the matching URL:

| CUDA Version | Install Command |
|-------------|-----------------|
| 12.4 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| 12.1 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| 11.8 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| CPU only | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |

---

### Step 4 — Install remaining packages

After PyTorch is installed and verified:
```bash
pip install -r requirements.txt
```

---

### Step 5 — Download the dataset

**Option A: Kaggle CLI (recommended)**

First get your API key:
- Go to kaggle.com → profile → Settings → API → Create New Token
- Downloads `kaggle.json`

Then:
```bash
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

cd /path/to/Conv_Project
mkdir data
kaggle datasets download -d paultimothymooney/kermany2018
unzip kermany2018.zip -d data/
```

**Option B: Manual download**

Go to: https://data.mendeley.com/datasets/rscbjbr9sj/3
Click Download, unzip into `Conv_Project/data/`

Your folder structure should end up as:
```
data/OCT2017/train/CNV/      ← training images
data/OCT2017/train/DME/
data/OCT2017/train/DRUSEN/
data/OCT2017/train/NORMAL/
data/OCT2017/test/           ← test images (968 total)
data/OCT2017/val/            ← official val (only 32 images, we resplit)
```

---

## Running Inference (diagnose a single image)

Make sure `oct_model.pth` is in the same folder as `inference.py`, then:

```bash
python inference.py path/to/scan.jpg
```

Example:
```bash
python inference.py data/OCT2017/test/CNV/CNV-1016042-1.jpeg
```

Expected output:
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
==================================================
```

---

## Running Training

Only needed if you want to retrain from scratch.
Training takes approximately 45 minutes on RTX 5080, longer on CPU.

```bash
python oct_train.py
```

Training will print progress each epoch:
```
Epoch 1/10 [Train]: 100%|██████████| 2088/2088 [04:15, loss=0.0176, acc=0.9286]
Epoch 1/10 [Val]:   100%|██████████|  522/522  [01:01, loss=0.1524, acc=0.9474]
Epoch 1/10 Summary | Train acc: 0.9290 | Val acc: 0.9478 | ...
```

Saved weights will be written to `oct_model.pth` when training finishes.

---

## MacBook M5 Pro Notes

- Uses Apple's MPS backend instead of CUDA — automatically detected
- Training is slower than RTX 5080 but works correctly
- One code change needed in oct_train.py if running on Mac:

Find this line:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Replace with:
```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Architecture | VGG16 |
| Pretrained on | ImageNet (IMAGENET1K_V1) |
| Frozen layers | features.0 through features.23 (blocks 1-4) |
| Unfrozen layers | features.24-28 (block5) |
| New head | nn.Linear(4096, 4) |
| Trainable params | 7,095,812 / 134,276,932 |
| Optimizer | Adam |
| Classifier lr | 0.001 |
| Features lr | 0.0001 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Epochs | 10 |
| Batch size | 32 |
| Train/Val split | 80/20 from training folder |

---

## Classes

| Label | Full Name | Description |
|-------|-----------|-------------|
| CNV | Choroidal Neovascularization | Abnormal blood vessel growth beneath retina |
| DME | Diabetic Macular Edema | Fluid accumulation in the macula |
| DRUSEN | Drusen | Deposits under retina, early macular degeneration sign |
| NORMAL | Normal | Healthy retina, no pathology detected |

---

## References

- Kermany et al. (2018). Identifying Medical Diagnoses and Treatable Diseases
  by Image-Based Deep Learning. Cell, 172(5), 1122-1131.
  https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

- Kaggle notebook (Mooney):
  https://www.kaggle.com/code/paultimothymooney/detect-retina-damage-from-oct-images

- Published paper using same approach (VGG16 + transfer learning on OCT):
  https://link.springer.com/article/10.1007/s42452-025-06565-6
  (Their result: 95.19% — this project achieved 99.48%)