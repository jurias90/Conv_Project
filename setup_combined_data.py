# setup_combined_data.py
# Builds combined_data/ folder from OCT dataset + all NOT_OCT sources
#
# Folder structure:
# Conv_Project/
# ├── data/
# │   ├── OCT2017/train/{CNV,DME,DRUSEN,NORMAL}/
# │   └── not_oct_data/
# │       ├── chest_xray/
# │       ├── cifar10/
# │       ├── fundus/
# │       ├── skin/
# │       ├── imagenet_mini/
# │       └── coco/
# └── combined_data/        ← built by this script, stays at project root
#
# Run: python setup_combined_data.py
# Safe to rerun — skips folders that already have images

import os
import random
import shutil
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURATION — update BASE for your machine
# Linux:  /home/dantails/PycharmProjects/Conv_Project
# Mac:    /Users/jesusurias/PycharmProjects/Conv_Project
# ─────────────────────────────────────────
BASE = "/home/dantails/PycharmProjects/Conv_Project"

# Source paths
OCT_TRAIN    = f"{BASE}/data/OCT2017/train"
NOT_OCT_BASE = f"{BASE}/data/not_oct_data"

# Destination — stays at project root, models already point here
COMBINED     = f"{BASE}/combined_data"

# How many images to sample from each NOT_OCT source
# None = use all available images from that source
SAMPLES_PER_SOURCE = {
    'chest_xray':    5000,
    'cifar10':       5000,
    'fundus':        None,   # ~3,662 — use all
    'skin':          5000,
    'imagenet_mini': 5000,
    'coco':          5000,
}

# Supported image extensions
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def find_images(folder):
    """Recursively find all image files — skips Mac metadata."""
    images = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs
                   if d not in ('__MACOSX', '.DS_Store')]
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                images.append(os.path.join(root, f))
    return images


def copy_images(sources, dest, n=None, label=""):
    """Copy n random images from sources to dest with unique names."""
    os.makedirs(dest, exist_ok=True)
    existing = len([f for f in os.listdir(dest)
                    if Path(f).suffix.lower() in IMAGE_EXTS])

    if existing > 0:
        print(f"  {label}: {existing} images already exist — skipping")
        return existing

    if n is not None and n < len(sources):
        selected = random.sample(sources, n)
    else:
        selected = sources

    copied = 0
    for src in selected:
        ext = Path(src).suffix.lower()
        dst = os.path.join(dest, f"{label}_{copied:06d}{ext}")
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception as e:
            print(f"    Warning: could not copy {src}: {e}")

    return copied


def main():
    random.seed(42)  # reproducible sampling

    print("=" * 65)
    print("OCT Combined Dataset Setup")
    print("=" * 65)

    # ─────────────────────────────────────────
    # PREFLIGHT: verify source paths exist
    # ─────────────────────────────────────────
    print("\nChecking source paths...")

    if not os.path.exists(OCT_TRAIN):
        print(f"  ✗ OCT2017/train NOT FOUND: {OCT_TRAIN}")
        print("  Download from: https://data.mendeley.com/datasets/rscbjbr9sj/3")
        return
    else:
        print(f"  ✓ OCT2017/train found")

    print("\nChecking NOT_OCT sources:")
    for source in SAMPLES_PER_SOURCE:
        src_path = f"{NOT_OCT_BASE}/{source}"
        if os.path.exists(src_path):
            images = find_images(src_path)
            print(f"  ✓ {source}: {len(images):,} images available")
        else:
            print(f"  ✗ {source}: not found — will skip")

    # ─────────────────────────────────────────
    # STEP 1: Create folder structure
    # ─────────────────────────────────────────
    print(f"\nBuilding: {COMBINED}")
    oct_classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    for cls in oct_classes + ['NOT_OCT']:
        os.makedirs(f"{COMBINED}/{cls}", exist_ok=True)
    print("✓ Folder structure created")

    # ─────────────────────────────────────────
    # STEP 2: Copy OCT training images
    # ─────────────────────────────────────────
    print("\nCopying OCT training images...")
    for cls in oct_classes:
        src_dir = f"{OCT_TRAIN}/{cls}"
        dst_dir = f"{COMBINED}/{cls}"

        if not os.path.exists(src_dir):
            print(f"  WARNING: {src_dir} not found — skipping")
            continue

        existing = len([f for f in os.listdir(dst_dir)
                        if Path(f).suffix.lower() in IMAGE_EXTS])
        if existing > 0:
            print(f"  {cls}: {existing:,} images already exist — skipping")
            continue

        images = find_images(src_dir)
        copied = 0
        for img in images:
            dst = os.path.join(dst_dir, Path(img).name)
            if os.path.exists(dst):
                stem = Path(img).stem
                ext  = Path(img).suffix
                dst  = os.path.join(dst_dir, f"{stem}_{copied}{ext}")
            try:
                shutil.copy2(img, dst)
                copied += 1
            except Exception as e:
                print(f"    Warning: {e}")

        print(f"  {cls}: copied {copied:,} images")

    # ─────────────────────────────────────────
    # STEP 3: Sample NOT_OCT from each source
    # ─────────────────────────────────────────
    print("\nSampling NOT_OCT images from each source...")
    not_oct_dst = f"{COMBINED}/NOT_OCT"
    total_not_oct = 0

    for source, n_samples in SAMPLES_PER_SOURCE.items():
        src_dir = f"{NOT_OCT_BASE}/{source}"

        if not os.path.exists(src_dir):
            print(f"  {source}: not found — skipping")
            continue

        images = find_images(src_dir)
        if len(images) == 0:
            print(f"  {source}: no images found — skipping")
            continue

        # Check if already copied from this source
        existing = len([f for f in os.listdir(not_oct_dst)
                        if f.startswith(source)])
        if existing > 0:
            print(f"  {source}: {existing:,} already copied — skipping")
            total_not_oct += existing
            continue

        sample_str = str(n_samples) if n_samples else f"all {len(images):,}"
        print(f"  {source}: {len(images):,} available "
              f"— sampling {sample_str}...")

        copied = copy_images(images, not_oct_dst,
                            n=n_samples, label=source)
        total_not_oct += copied
        print(f"    → {copied:,} images added")

    # ─────────────────────────────────────────
    # STEP 4: Final summary
    # ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL DATASET SUMMARY")
    print("=" * 65)

    counts    = {}
    total     = 0
    max_count = 1

    for cls in oct_classes + ['NOT_OCT']:
        cls_dir = f"{COMBINED}/{cls}"
        if os.path.exists(cls_dir):
            count = len([f for f in os.listdir(cls_dir)
                         if Path(f).suffix.lower() in IMAGE_EXTS])
            counts[cls] = count
            total += count
            max_count = max(max_count, count)

    for cls, count in counts.items():
        bar = '█' * int((count / max_count) * 30)
        print(f"  {cls:<12} {count:>8,}  {bar}")

    print(f"  {'─'*52}")
    print(f"  {'TOTAL':<12} {total:>8,}")
    print("=" * 65)

    # Class balance check
    if counts:
        min_cls = min(counts, key=counts.get)
        max_cls = max(counts, key=counts.get)
        ratio   = counts[max_cls] / max(counts[min_cls], 1)

        if ratio > 5:
            print(f"\n⚠  Class imbalance warning:")
            print(f"   Largest:  {max_cls} ({counts[max_cls]:,})")
            print(f"   Smallest: {min_cls} ({counts[min_cls]:,})")
            print(f"   Ratio: {ratio:.1f}x — consider reducing "
                  f"{max_cls} in SAMPLES_PER_SOURCE above")
        else:
            print(f"\n✓ Class balance OK (ratio: {ratio:.1f}x)")

    print("\n✓ Setup complete — ready to train")
    print("\nRun models:")
    print("  python models/alexnet_5class_train.py")
    print("  python models/alexnet_5class_scratch.py")
    print("  python models/vgg16_5class_train.py")
    print("  python models/resnet50_5class_train.py")


if __name__ == '__main__':
    main()