import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import sys
import ssl
import certifi
import os

# Mac SSL fix — works even when PyTorch uses its own downloader
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()


# ─────────────────────────────────────────
# SECTION 1: What Device am I even Using
# ─────────────────────────────────────────
def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

# ─────────────────────────────────────────
# SECTION 2: Data loading
# ─────────────────────────────────────────
    data_dir = "./data/OCT2017 "

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    full_train = datasets.ImageFolder(data_dir + "/train", transform=transform)
    test_data = datasets.ImageFolder(data_dir + "/test", transform=transform)

    train_size = int(0.8 * len(full_train))
    val_size = len(full_train)-train_size

    train_data,val_data = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size= 32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size= 32, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size= 32, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train size: {len(train_data)} images")
    print(f"Val size: {len(val_data)} images")
    print(f"Test size: {len(test_data)} images")
    print(f"Classes: {full_train.classes}")

# ─────────────────────────────────────────
# SECTION 3: Model setup
# ─────────────────────────────────────────
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if 'layer4' in name:
            param.requires_grad = True


    model.fc = nn.Linear(2048, 10)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable parameters: {trainable:,} / {total:,}")
    print(f"Model ready on: {next(model.parameters()).device}")
    # ─────────────────────────────────────────
    # SECTION 4: Training loop
    # ─────────────────────────────────────────

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam([
        {'params': model.fc.parameters(), 'lr': 0.001},
        {'params': filter(lambda p: p.requires_grad,
                          model.layer4.parameters()), 'lr': 0.0001}
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
                         desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]",
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
                'acc': f"{train_correct / ((train_bar.n + 1) * 32):.4f}"
            })

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0

        val_bar = tqdm(val_loader,
                       desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]  ",
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
                    'acc': f"{val_correct / ((val_bar.n + 1) * 32):.4f}"
                })

        print(f"\nEpoch {epoch + 1}/{EPOCHS} Summary | "
              f"Train acc: {train_correct / len(train_data):.4f} | "
              f"Val acc: {val_correct / len(val_data):.4f} | "
              f"Train loss: {train_loss / len(train_loader):.4f} | "
              f"Val loss: {val_loss / len(val_loader):.4f}\n",
              flush=True)

        scheduler.step(val_loss / len(val_loader))

    # ─────────────────────────────────────────
    # SECTION 5: Evaluation
    # ─────────────────────────────────────────

    model.eval()
    test_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_correct += (outputs.argmax(1) == labels).sum().item()

    final_acc = test_correct / len(test_data)
    print(f"\nFinal test accuracy: {final_acc:.4f}")
    print(f"Correct: {test_correct}/{len(test_data)}")

    # Save with different name so VGG16 weights aren't overwritten
    torch.save(model.state_dict(), 'resnet50_model.pth')
    print("ResNet50 model saved to resnet50_model.pth")
    

if __name__ == "__main__":
    main()