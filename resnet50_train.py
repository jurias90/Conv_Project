import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import sys


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
    

if __name__ == "__main__":
    main()