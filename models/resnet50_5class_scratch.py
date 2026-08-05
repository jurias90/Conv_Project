import os
import json
import ssl
import certifi
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet50
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

BASE     = "/home/dantails/PycharmProjects/Conv_Project"
DATA_DIR = f"{BASE}/combined_data"
WEIGHTS  = f"{BASE}/weights/resnet50_5class_scratch.pth"
CM_PATH  = f"{BASE}/results/confusion_matrix_resnet50_5class_scratch.png"
LOG_PATH = f"{BASE}/results/resnet50_5class_scratch_log.json"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("MODE: Training from SCRATCH (no pretrained weights)")

    # ─────────────────────────────────────────
    # SECTION 2: Data loading
    # ─────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    full_data  = datasets.ImageFolder(DATA_DIR, transform=transform)
    test_size  = int(0.1 * len(full_data))
    train_val  = len(full_data) - test_size
    train_size = int(0.8 * train_val)
    val_size   = train_val - train_size

    train_data, val_data, test_data = random_split(
        full_data, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_data, batch_size=32,
                              shuffle=True,  num_workers=0,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_data,   batch_size=32,
                              shuffle=False, num_workers=0,
                              pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_data,  batch_size=32,
                              shuffle=False, num_workers=0,
                              pin_memory=torch.cuda.is_available())

    print(f"Train: {len(train_data)} images")
    print(f"Val:   {len(val_data)} images")
    print(f"Test:  {len(test_data)} images")
    print(f"Classes: {full_data.classes}")

    # ─────────────────────────────────────────
    # SECTION 3: Model setup — NO pretrained weights
    # ─────────────────────────────────────────
    model = resnet50(weights=None)  # ← random initialization

    # All layers trainable — nothing to freeze
    model.fc = nn.Linear(2048, 5)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")
    print("NOTE: All params trainable — starting from random weights")

    # ─────────────────────────────────────────
    # SECTION 4: Training loop — 20 epochs
    # ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    EPOCHS = 20

    training_log = {
        'train_acc':  [],
        'val_acc':    [],
        'train_loss': [],
        'val_loss':   []
    }

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0

        train_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch+1}/{EPOCHS} [Train]",
                         leave=True)

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc':  f"{train_correct/((train_bar.n+1)*32):.4f}"
            })

        model.eval()
        val_loss = 0
        val_correct = 0

        val_bar = tqdm(val_loader,
                       desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  ",
                       leave=True)

        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss    += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc':  f"{val_correct/((val_bar.n+1)*32):.4f}"
                })

        epoch_train_acc  = train_correct/len(train_data)
        epoch_val_acc    = val_correct/len(val_data)
        epoch_train_loss = train_loss/len(train_loader)
        epoch_val_loss   = val_loss/len(val_loader)

        training_log['train_acc'].append(epoch_train_acc)
        training_log['val_acc'].append(epoch_val_acc)
        training_log['train_loss'].append(epoch_train_loss)
        training_log['val_loss'].append(epoch_val_loss)

        print(f"\nEpoch {epoch+1}/{EPOCHS} Summary | "
              f"Train acc: {epoch_train_acc:.4f} | "
              f"Val acc: {epoch_val_acc:.4f} | "
              f"Train loss: {epoch_train_loss:.4f} | "
              f"Val loss: {epoch_val_loss:.4f}\n",
              flush=True)

        scheduler.step(epoch_val_loss)

    with open(LOG_PATH, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to {LOG_PATH}")

    # ─────────────────────────────────────────
    # SECTION 5: Evaluation + Confusion Matrix
    # ─────────────────────────────────────────
    model.eval()
    test_correct = 0
    all_preds    = []
    all_labels   = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds   = outputs.argmax(1)
            test_correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    final_acc = test_correct / len(test_data)
    print(f"\nFinal test accuracy: {final_acc:.4f}")
    print(f"Correct: {test_correct}/{len(test_data)}")

    classes = full_data.classes
    print("\nPer-class accuracy:")
    for i, cls in enumerate(classes):
        cls_correct = sum(p == l for p, l in
                         zip(all_preds, all_labels) if l == i)
        cls_total   = sum(1 for l in all_labels if l == i)
        if cls_total > 0:
            print(f"  {cls:10} {cls_correct}/{cls_total} "
                  f"= {cls_correct/cls_total:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title=f'ResNet50 5-class FROM SCRATCH — {final_acc:.2%}\n'
                 f'No pretrained weights — compare to transfer learning',
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2
                    else "black")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    print(f"Confusion matrix saved to {CM_PATH}")

    torch.save(model.state_dict(), WEIGHTS)
    print(f"Weights saved to {WEIGHTS}")


if __name__ == '__main__':
    main()