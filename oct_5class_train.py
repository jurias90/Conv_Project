import os
import ssl
import certifi
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torchvision.models import VGG16_Weights
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# SSL fix for Mac compatibility
ssl.create_default_context = lambda *args, **kwargs: \
    ssl.create_default_context(*args, cafile=certifi.where(), **kwargs)

def main():
    # ─────────────────────────────────────────
    # SECTION 1: Device setup
    # ─────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ─────────────────────────────────────────
    # SECTION 2: Data loading
    # ─────────────────────────────────────────
    data_dir = "/home/dantails/PycharmProjects/Conv_Project/combined_data"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    full_train = datasets.ImageFolder(data_dir, transform=transform)
    test_size  = int(0.1 * len(full_train))
    train_val  = len(full_train) - test_size
    train_size = int(0.8 * train_val)
    val_size   = train_val - train_size

    train_data, val_data, test_data = random_split(
        full_train, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"Train: {len(train_data)} images")
    print(f"Val:   {len(val_data)} images")
    print(f"Test:  {len(test_data)} images")
    print(f"Classes: {full_train.classes}")

    # ─────────────────────────────────────────
    # SECTION 3: Model setup
    # ─────────────────────────────────────────
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze block5
    for name, param in model.named_parameters():
        if 'features.24' in name or \
           'features.26' in name or \
           'features.28' in name:
            param.requires_grad = True

    # 5 classes instead of 4
    model.classifier[6] = nn.Linear(4096, 5)

    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")
    print(f"Model ready on: {next(model.parameters()).device}")

    # ─────────────────────────────────────────
    # SECTION 4: Training loop
    # ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam([
        {'params': model.classifier[6].parameters(), 'lr': 0.001},
        {'params': filter(lambda p: p.requires_grad,
                          model.features.parameters()), 'lr': 0.0001}
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
    )

    EPOCHS = 10

    for epoch in range(EPOCHS):

        # Training phase
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

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc':  f"{train_correct / ((train_bar.n + 1) * 32):.4f}"
            })

        # Validation phase
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
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

                val_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc':  f"{val_correct / ((val_bar.n + 1) * 32):.4f}"
                })

        print(f"\nEpoch {epoch+1}/{EPOCHS} Summary | "
              f"Train acc: {train_correct/len(train_data):.4f} | "
              f"Val acc: {val_correct/len(val_data):.4f} | "
              f"Train loss: {train_loss/len(train_loader):.4f} | "
              f"Val loss: {val_loss/len(val_loader):.4f}\n",
              flush=True)

        scheduler.step(val_loss / len(val_loader))

    # ─────────────────────────────────────────
    # SECTION 5: Evaluation
    # ─────────────────────────────────────────
    model.eval()
    test_correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            test_correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    final_acc = test_correct / len(test_data)
    print(f"\nFinal test accuracy: {final_acc:.4f}")
    print(f"Correct: {test_correct}/{len(test_data)}")

    # Per-class accuracy breakdown
    print("\nPer-class accuracy:")
    classes = full_train.classes
    for i, cls in enumerate(classes):
        cls_mask = [l == i for l in all_labels]
        cls_correct = sum(p == l for p, l in zip(all_preds, all_labels) if l == i)
        cls_total = sum(cls_mask)
        if cls_total > 0:
            print(f"  {cls:10} {cls_correct}/{cls_total} = {cls_correct/cls_total:.4f}")

    # Save model
    torch.save(model.state_dict(), 'oct_5class_model.pth')
    print("\n5-class model saved to oct_5class_model.pth")


if __name__ == '__main__':
    main()