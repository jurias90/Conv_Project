# generate_transfer_logs.py
# Creates training logs from already-completed transfer learning runs
# Run this ONCE to generate the JSON files needed for comparison plots
# Based on actual epoch data recorded during training
import json
import os
BASE = "/home/dantails/PycharmProjects/Conv_Project/results"
os.makedirs(BASE, exist_ok=True)


# ─────────────────────────────────────────
# VGG16 5-class transfer learning
# Actual epoch data from training run
# ─────────────────────────────────────────
vgg16_transfer = {
    'train_acc':
        [0.9290, 0.9600,
         0.9702, 0.9790,
         0.9840, 0.9870,
         0.9931, 0.9944,
         0.9944, 0.9959],
    'val_acc':
        [0.9478, 0.9515,
         0.9545, 0.9559,
         0.9607, 0.9622,
         0.9644, 0.9624,
         0.9647, 0.9663],
    'train_loss': [0.2147, 0.1234,
                   0.0899, 0.0621,
                   0.0464, 0.0376,
                   0.0181, 0.0146,
                   0.0142, 0.0078],
    'val_loss':   [0.1575, 0.1418,
                   0.1323, 0.1395,
                   0.1376, 0.1425,
                   0.1642, 0.1792,
                   0.1628, 0.1794] }
path = f"{BASE}/vgg16_5class_transfer_log.json"
with open(path, 'w') as f:
    json.dump(vgg16_transfer, f, indent=2)
print(f"Saved: {path}")


# ─────────────────────────────────────────
# ResNet50 5-class transfer learning
# Actual epoch data from training run
# ─────────────────────────────────────────
resnet50_transfer = {
    'train_acc':  [0.9321, 0.9608,
                   0.9730, 0.9816,
                   0.9861, 0.9877,
                   0.9938, 0.9954,
                   0.9956, 0.9966],
    'val_acc':    [0.9537, 0.9555,
                   0.9638, 0.9645,
                   0.9638, 0.9604,
                   0.9650, 0.9644,
                   0.9653, 0.9676],
    'train_loss': [0.2028, 0.1160,
                   0.0790, 0.0542,
                   0.0410, 0.0381,
                   0.0172, 0.0125,
                   0.0124, 0.0076],
    'val_loss':   [0.1347, 0.1396,
                   0.1165, 0.1202,
                   0.1414, 0.1584,
                   0.1662, 0.1858,
                   0.1862, 0.2031] }

path = f"{BASE}/resnet50_5class_transfer_log.json"
with open(path, 'w') as f:
    json.dump(resnet50_transfer, f, indent=2)

print(f"Saved: {path}")
print("\nTransfer learning logs generated successfully.")
print("Now run the scratch models, then run compare_all_models.py")