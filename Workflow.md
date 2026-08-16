# Complete Training & Comparison Workflow
# Run in this exact order after new datasets are downloaded

# ─────────────────────────────────────────
# STEP 1 — DONE
# not_oct_data moved under data/
# ─────────────────────────────────────────

# ─────────────────────────────────────────
# STEP 2 — Wipe old combined_data
# ─────────────────────────────────────────
rm -rf ~/PycharmProjects/Conv_Project/combined_data

# ─────────────────────────────────────────
# STEP 3 — Rebuild combined_data with all
#           new datasets included
# ─────────────────────────────────────────
cd ~/PycharmProjects/Conv_Project
python setup_combined_data.py

# Verify counts before training:
python3 -c "
import os
from pathlib import Path
base = 'combined_data'
exts = {'.jpg','.jpeg','.png','.bmp','.tiff'}
total = 0
for cls in ['CNV','DME','DRUSEN','NORMAL','NOT_OCT']:
    d = f'{base}/{cls}'
    if os.path.exists(d):
        n = len([f for f in os.listdir(d)
                 if Path(f).suffix.lower() in exts])
        total += n
        print(f'{cls:<12} {n:>8,}')
print(f'{\"TOTAL\":<12} {total:>8,}')
"

# ─────────────────────────────────────────
# STEP 4 — Train all models
#
# Terminal 1: transfer learning (all 5)
# Terminal 2: from scratch (3 models)
# Terminal 3: 4-class models (both)
#
# Start all three terminals simultaneously
# ─────────────────────────────────────────

# TERMINAL 1 — Transfer learning (5-class)
# ~15 min + ~55 min + ~55 min = ~2hr 5min
python models/alexnet_5class_train.py && \
python models/vgg16_5class_train.py && \
python models/resnet50_5class_train.py

# TERMINAL 2 — From scratch (5-class)
# ~25 min + ~2hr + ~2hr = ~4hr 25min
python models/alexnet_5class_scratch.py && \
python models/vgg16_5class_scratch.py && \
python models/resnet50_5class_scratch.py

# TERMINAL 3 — 4-class baselines
# ~45 min + ~25 min = ~1hr 10min
python models/vgg16_4class_train.py && \
python models/resnet50_4class_train.py

# ─────────────────────────────────────────
# STEP 5 — Generate all plots + summary
# Run AFTER all terminals finish
# ─────────────────────────────────────────
python compare_all_models.py

# ─────────────────────────────────────────
# STEP 6 — Push everything to GitHub
# ─────────────────────────────────────────
git add models/
git add results/
git add weights/
git add compare_all_models.py
git add setup_combined_data.py
git commit -m "Retrain all models with expanded NOT_OCT dataset"
git push

# ─────────────────────────────────────────
# EXPECTED RESULTS after retraining
# ─────────────────────────────────────────
# Files auto-saved to results/:
#
#   Training logs (JSON):
#     alexnet_5class_transfer_log.json
#     alexnet_5class_scratch_log.json
#     vgg16_5class_transfer_log.json
#     vgg16_5class_scratch_log.json
#     resnet50_5class_transfer_log.json
#     resnet50_5class_scratch_log.json
#     vgg16_4class_transfer_log.json
#     resnet50_4class_transfer_log.json
#
#   Confusion matrices (PNG):
#     confusion_matrix_alexnet_5class_transfer.png
#     confusion_matrix_alexnet_5class_scratch.png
#     confusion_matrix_vgg16_5class.png
#     confusion_matrix_vgg16_5class_scratch.png
#     confusion_matrix_resnet50_5class.png
#     confusion_matrix_resnet50_5class_scratch.png
#     confusion_matrix_vgg16_4class.png
#     confusion_matrix_resnet50_4class.png
#
#   Comparison plots (PNG):
#     compare_all_training_curves.png
#     compare_alexnet_transfer_vs_scratch.png
#     compare_vgg16_transfer_vs_scratch.png
#     compare_resnet50_transfer_vs_scratch.png
#     compare_final_accuracy_all.png
#
#   Weights (pth) saved to weights/:
#     alexnet_5class_transfer.pth
#     alexnet_5class_scratch.pth
#     vgg16_5class.pth
#     vgg16_5class_scratch.pth
#     resnet50_5class.pth
#     resnet50_5class_scratch.pth
#     vgg16_4class.pth
#     resnet50_4class.pth