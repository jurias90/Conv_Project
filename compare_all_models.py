# compare_all_models.py
# Generates all comparison plots showing transfer learning vs scratch
# Run AFTER all four training logs exist in results/

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE = "/home/dantails/PycharmProjects/Conv_Project"
RESULTS = f"{BASE}/results"


def load_log(filename):
    path = f"{RESULTS}/{filename}"
    if not os.path.exists(path):
        print(f"WARNING: {filename} not found — run training first")
        return None
    with open(path) as f:
        return json.load(f)


def main():

    # Load all logs
    vgg16_tl     = load_log("vgg16_5class_transfer_log.json")
    resnet50_tl  = load_log("resnet50_5class_transfer_log.json")
    vgg16_sc     = load_log("vgg16_5class_scratch_log.json")
    resnet50_sc  = load_log("resnet50_5class_scratch_log.json")

    # ─────────────────────────────────────────
    # PLOT 1: Transfer Learning vs Scratch
    # VGG16 side by side
    # ─────────────────────────────────────────
    if vgg16_tl and vgg16_sc:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        tl_epochs = list(range(1, len(vgg16_tl['val_acc']) + 1))
        sc_epochs = list(range(1, len(vgg16_sc['val_acc']) + 1))

        # Val accuracy
        axes[0].plot(tl_epochs, vgg16_tl['val_acc'],
                    'b-o', label='Transfer Learning', linewidth=2.5)
        axes[0].plot(sc_epochs, vgg16_sc['val_acc'],
                    'r--s', label='From Scratch', linewidth=2.5)
        axes[0].axhline(y=0.966, color='orange', linestyle=':',
                       linewidth=1.5, label='Kermany 2018 (96.6%)')
        axes[0].set_title('Validation Accuracy', fontweight='bold',
                          fontsize=13)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Accuracy', fontsize=11)
        axes[0].set_ylim(0.15, 1.05)
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)

        # Val loss
        axes[1].plot(tl_epochs, vgg16_tl['val_loss'],
                    'b-o', label='Transfer Learning', linewidth=2.5)
        axes[1].plot(sc_epochs, vgg16_sc['val_loss'],
                    'r--s', label='From Scratch', linewidth=2.5)
        axes[1].set_title('Validation Loss', fontweight='bold',
                          fontsize=13)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Loss', fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)

        fig.suptitle(
            'VGG16 5-class: Transfer Learning vs Training from Scratch\n'
            'Same architecture, same data, same training setup',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        out = f"{RESULTS}/compare_vgg16_transfer_vs_scratch.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved: {out}")

    # ─────────────────────────────────────────
    # PLOT 2: Transfer Learning vs Scratch
    # ResNet50 side by side
    # ─────────────────────────────────────────
    if resnet50_tl and resnet50_sc:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        tl_epochs = list(range(1, len(resnet50_tl['val_acc']) + 1))
        sc_epochs = list(range(1, len(resnet50_sc['val_acc']) + 1))

        axes[0].plot(tl_epochs, resnet50_tl['val_acc'],
                    'g-o', label='Transfer Learning', linewidth=2.5)
        axes[0].plot(sc_epochs, resnet50_sc['val_acc'],
                    'r--s', label='From Scratch', linewidth=2.5)
        axes[0].axhline(y=0.966, color='orange', linestyle=':',
                       linewidth=1.5, label='Kermany 2018 (96.6%)')
        axes[0].set_title('Validation Accuracy', fontweight='bold',
                          fontsize=13)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Accuracy', fontsize=11)
        axes[0].set_ylim(0.15, 1.05)
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)

        axes[1].plot(tl_epochs, resnet50_tl['val_loss'],
                    'g-o', label='Transfer Learning', linewidth=2.5)
        axes[1].plot(sc_epochs, resnet50_sc['val_loss'],
                    'r--s', label='From Scratch', linewidth=2.5)
        axes[1].set_title('Validation Loss', fontweight='bold',
                          fontsize=13)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Loss', fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)

        fig.suptitle(
            'ResNet50 5-class: Transfer Learning vs Training from Scratch\n'
            'Same architecture, same data, same training setup',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        out = f"{RESULTS}/compare_resnet50_transfer_vs_scratch.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved: {out}")

    # ─────────────────────────────────────────
    # PLOT 3: All 6 models — val accuracy
    # The poster's main comparison graph
    # ─────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Colors and styles
    styles = {
        'vgg16_tl': ('b', '-', 'o', 'VGG16 Transfer Learning'),
        'resnet50_tl': ('g', '-', '^', 'ResNet50 Transfer Learning'),
        'vgg16_sc': ('b', '--', 's', 'VGG16 From Scratch'),
        'resnet50_sc': ('g', '--', 'D', 'ResNet50 From Scratch'),
    }

    logs = {
        'vgg16_tl': vgg16_tl,
        'resnet50_tl': resnet50_tl,
        'vgg16_sc': vgg16_sc,
        'resnet50_sc': resnet50_sc,
    }

    for key, log in logs.items():
        if log is None:
            continue
        color, linestyle, marker, label = styles[key]
        epochs = list(range(1, len(log['val_acc']) + 1))

        ax1.plot(epochs, log['val_acc'],
                 color=color, linestyle=linestyle,
                 marker=marker, label=label,
                 linewidth=2.5, markersize=6)

        ax2.plot(epochs, log['val_loss'],
                 color=color, linestyle=linestyle,
                 marker=marker, label=label,
                 linewidth=2.5, markersize=6)

    # Reference lines on accuracy plot
    ax1.axhline(y=0.966, color='orange', linestyle=':',
                linewidth=2, label='Kermany 2018 (96.6%)')
    ax1.axhline(y=0.9519, color='purple', linestyle=':',
                linewidth=2, label='Published VGG16 paper (95.19%)')

    ax1.set_title('Validation Accuracy Per Epoch — All Models',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Accuracy', fontsize=12)
    ax1.set_ylim(0.15, 1.05)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.axvline(x=10, color='gray', linestyle=':',
                linewidth=1, alpha=0.5,
                label='Transfer learning stops here')
    ax1.text(10.2, 0.2, 'Transfer\nlearning\nstops',
             fontsize=9, color='gray')

    ax2.set_title('Validation Loss Per Epoch — All Models',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.axvline(x=10, color='gray', linestyle=':',
                linewidth=1, alpha=0.5)

    fig.suptitle(
        'Transfer Learning vs Training from Scratch\n'
        'VGG16 and ResNet50 on OCT Retinal Classification (5-class)',
        fontsize=15, fontweight='bold'
    )

    plt.tight_layout()
    out = f"{RESULTS}/compare_all_training_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # ─────────────────────────────────────────
    # PLOT 4: Final accuracy bar chart
    # All models compared at a glance
    # ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))

    models = [
        'VGG16\n4-class\nTransfer',
        'ResNet50\n4-class\nTransfer',
        'VGG16\n5-class\nTransfer',
        'ResNet50\n5-class\nTransfer',
        'VGG16\n5-class\nScratch',
        'ResNet50\n5-class\nScratch',
    ]

    accuracies = [
        98.86,   # VGG16 4-class transfer
        99.69,   # ResNet50 4-class transfer
        97.07,   # VGG16 5-class transfer
        96.53,   # ResNet50 5-class transfer
        0,       # VGG16 5-class scratch (fill after training)
        0,       # ResNet50 5-class scratch (fill after training)
    ]

    colors = [
        '#1565c0',   # blue — transfer
        '#1565c0',   # blue — transfer
        '#1976d2',   # lighter blue — transfer 5-class
        '#1976d2',   # lighter blue — transfer 5-class
        '#d32f2f',   # red — scratch
        '#d32f2f',   # red — scratch
    ]

    bars = ax.bar(models, accuracies, color=colors,
                  edgecolor='white', width=0.6)

    for bar, acc in zip(bars, accuracies):
        if acc > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    acc + 0.3, f'{acc}%',
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=11)
        else:
            ax.text(bar.get_x() + bar.get_width()/2,
                    5, 'TBD',
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=11,
                    color='white')

    ax.axhline(y=96.6, color='orange', linestyle='--',
              linewidth=2, label='Kermany 2018 (96.6%)')
    ax.axhline(y=95.19, color='purple', linestyle='--',
              linewidth=2, label='Published VGG16 paper (95.19%)')

    blue_patch  = mpatches.Patch(color='#1565c0',
                                  label='Transfer Learning')
    red_patch   = mpatches.Patch(color='#d32f2f',
                                  label='From Scratch')
    ax.legend(handles=[blue_patch, red_patch,
                       plt.Line2D([0], [0], color='orange',
                                  linestyle='--', label='Kermany 2018'),
                       plt.Line2D([0], [0], color='purple',
                                  linestyle='--',
                                  label='Published VGG16 2025')],
              fontsize=10)

    ax.set_ylim(0, 105)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(
        'Final Test Accuracy — All Models\n'
        'Transfer Learning consistently outperforms training from scratch',
        fontsize=14, fontweight='bold'
    )
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = f"{RESULTS}/compare_final_accuracy_all.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    print("\nAll comparison plots generated.")
    print("Files saved to results/:")
    print("  compare_vgg16_transfer_vs_scratch.png")
    print("  compare_resnet50_transfer_vs_scratch.png")
    print("  compare_all_training_curves.png")
    print("  compare_final_accuracy_all.png")


if __name__ == '__main__':
    main()