# compare_all_models.py
# Automatically reads all JSON training logs from results/
# and generates all comparison plots.
#
# Fully automatic — no manual data entry needed.
# Just run after training completes:
#   python compare_all_models.py

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

BASE    = "/home/dantails/PycharmProjects/Conv_Project"
RESULTS = f"{BASE}/results"

# ─────────────────────────────────────────
# Display order for bar chart
# Add new models here if you add more architectures
# ─────────────────────────────────────────
DISPLAY_ORDER = [
    ('alexnet_5class_transfer_log',  'AlexNet\nTransfer'),
    ('vgg16_5class_transfer_log',    'VGG16\nTransfer'),
    ('resnet50_5class_transfer_log', 'ResNet50\nTransfer'),
    ('alexnet_5class_scratch_log',   'AlexNet\nScratch'),
    ('vgg16_5class_scratch_log',     'VGG16\nScratch'),
    ('resnet50_5class_scratch_log',  'ResNet50\nScratch'),
]

# Color per architecture — consistent across all plots
ARCH_COLORS = {
    'alexnet':  '#e67e22',  # orange
    'vgg16':    '#2980b9',  # blue
    'resnet50': '#27ae60',  # green
}

# Reference lines — published benchmarks
REFERENCES = [
    (0.966,  'orange',  'Kermany 2018 (96.6%)'),
    (0.9519, 'purple',  'Published VGG16 2025 (95.19%)'),
]


def load_all_logs():
    """Auto-discover and load all *_log.json files in results/"""
    logs = {}
    json_files = sorted(Path(RESULTS).glob("*_log.json"))
    if not json_files:
        print(f"No JSON logs found in {RESULTS}")
        return logs
    for path in json_files:
        with open(path) as f:
            data = json.load(f)
        logs[path.stem] = data
        epochs    = len(data.get('val_acc', []))
        final_acc = data.get('final_test_acc')
        acc_str   = f"{final_acc:.4f}" if final_acc else "no final_test_acc"
        print(f"  Loaded: {path.name} — {epochs} epochs — {acc_str}")
    return logs


def get_arch_color(key):
    """Get color based on architecture name in key."""
    for arch, color in ARCH_COLORS.items():
        if arch in key:
            return color
    return '#888888'


def get_linestyle(key):
    """Solid for transfer, dashed for scratch."""
    return '--' if 'scratch' in key else '-'


def get_marker(key):
    """Circle for transfer, square for scratch."""
    return 's' if 'scratch' in key else 'o'


def get_label(key):
    """Clean display label from log key."""
    labels = {
        'alexnet_5class_transfer_log':   'AlexNet Transfer',
        'alexnet_5class_scratch_log':    'AlexNet Scratch',
        'vgg16_5class_transfer_log':     'VGG16 Transfer',
        'vgg16_5class_scratch_log':      'VGG16 Scratch',
        'resnet50_5class_transfer_log':  'ResNet50 Transfer',
        'resnet50_5class_scratch_log':   'ResNet50 Scratch',
        'alexnet_4class_transfer_log':   'AlexNet 4-class Transfer',
        'vgg16_4class_transfer_log':     'VGG16 4-class Transfer',
        'resnet50_4class_transfer_log':  'ResNet50 4-class Transfer',
    }
    return labels.get(key, key.replace('_log', '')
                              .replace('_', ' ').title())


def get_final_acc(log):
    """
    Get final test accuracy from log.
    Tries final_test_acc first (auto-saved by new training scripts).
    Falls back to best val_acc if not present (old runs).
    """
    if 'final_test_acc' in log:
        return log['final_test_acc'], 'test'
    if log.get('val_acc'):
        return max(log['val_acc']), 'val_best'
    return None, None


def main():
    print("=" * 60)
    print("Loading training logs...")
    print("=" * 60)
    logs = load_all_logs()

    if not logs:
        print("No logs found — run training scripts first.")
        return

    os.makedirs(RESULTS, exist_ok=True)

    # Separate transfer vs scratch for targeted plots
    transfer_keys = [k for k in logs if 'transfer' in k]
    scratch_keys  = [k for k in logs if 'scratch'  in k]

    print(f"\nTransfer learning logs: {len(transfer_keys)}")
    print(f"From scratch logs:      {len(scratch_keys)}")

    # ─────────────────────────────────────────
    # PLOT 1: All models — val accuracy + loss
    # ─────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    fig.patch.set_facecolor('white')

    for key, log in sorted(logs.items()):
        if 'val_acc' not in log or not log['val_acc']:
            continue
        color  = get_arch_color(key)
        ls     = get_linestyle(key)
        marker = get_marker(key)
        label  = get_label(key)
        alpha  = 0.6 if 'scratch' in key else 1.0
        epochs = list(range(1, len(log['val_acc']) + 1))

        ax1.plot(epochs, log['val_acc'],
                color=color, linestyle=ls,
                marker=marker, label=label,
                linewidth=2, markersize=5, alpha=alpha)

    for y, color, label in REFERENCES:
        ax1.axhline(y=y, color=color, linestyle=':',
                   linewidth=2, label=label)

    # Vertical line where transfer learning stops
    ax1.axvline(x=10, color='gray', linestyle=':',
               linewidth=1.5, alpha=0.5)
    ax1.text(10.2, 0.20,
             'Transfer\nlearning\nstops\n(epoch 10)',
             fontsize=8, color='gray')

    ax1.set_title('Validation Accuracy Per Epoch — All Models',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Accuracy', fontsize=12)
    ax1.set_ylim(0.15, 1.05)
    ax1.legend(fontsize=9, loc='lower right', ncol=2)
    ax1.grid(alpha=0.3)

    for key, log in sorted(logs.items()):
        if 'val_loss' not in log or not log['val_loss']:
            continue
        color  = get_arch_color(key)
        ls     = get_linestyle(key)
        marker = get_marker(key)
        label  = get_label(key)
        alpha  = 0.6 if 'scratch' in key else 1.0
        epochs = list(range(1, len(log['val_loss']) + 1))

        ax2.plot(epochs, log['val_loss'],
                color=color, linestyle=ls,
                marker=marker, label=label,
                linewidth=2, markersize=5, alpha=alpha)

    ax2.axvline(x=10, color='gray', linestyle=':',
               linewidth=1.5, alpha=0.5)
    ax2.set_title('Validation Loss Per Epoch — All Models',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.legend(fontsize=9, loc='upper right', ncol=2)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        'Transfer Learning vs From Scratch\n'
        'AlexNet, VGG16, ResNet50 — OCT Retinal Classification',
        fontsize=15, fontweight='bold'
    )
    plt.tight_layout()
    out = f"{RESULTS}/compare_all_training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")

    # ─────────────────────────────────────────
    # PLOT 2: Per-architecture transfer vs scratch
    # ─────────────────────────────────────────
    for arch in ['alexnet', 'vgg16', 'resnet50']:
        tl_key = f"{arch}_5class_transfer_log"
        sc_key = f"{arch}_5class_scratch_log"

        if tl_key not in logs and sc_key not in logs:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        color = ARCH_COLORS.get(arch, '#888888')

        if tl_key in logs:
            tl  = logs[tl_key]
            ep  = list(range(1, len(tl['val_acc']) + 1))
            acc, src = get_final_acc(tl)
            lbl = f"Transfer Learning ({acc:.2%} {src})" \
                  if acc else "Transfer Learning"
            ax1.plot(ep, tl['val_acc'], color=color,
                    linestyle='-', marker='o',
                    label=lbl, linewidth=2.5)
            ax2.plot(ep, tl['val_loss'], color=color,
                    linestyle='-', marker='o',
                    label='Transfer Learning', linewidth=2.5)

        if sc_key in logs:
            sc  = logs[sc_key]
            ep  = list(range(1, len(sc['val_acc']) + 1))
            acc, src = get_final_acc(sc)
            lbl = f"From Scratch ({acc:.2%} {src})" \
                  if acc else "From Scratch"
            ax1.plot(ep, sc['val_acc'], color=color,
                    linestyle='--', marker='s',
                    label=lbl, linewidth=2.5, alpha=0.7)
            ax2.plot(ep, sc['val_loss'], color=color,
                    linestyle='--', marker='s',
                    label='From Scratch', linewidth=2.5, alpha=0.7)

        for y, ref_color, ref_label in REFERENCES:
            ax1.axhline(y=y, color=ref_color, linestyle=':',
                       linewidth=1.5, label=ref_label)

        ax1.set_title('Validation Accuracy',
                      fontweight='bold', fontsize=13)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Accuracy', fontsize=11)
        ax1.set_ylim(0.15, 1.05)
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        ax2.set_title('Validation Loss',
                      fontweight='bold', fontsize=13)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Loss', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        fig.suptitle(
            f'{arch.upper()} — Transfer Learning vs From Scratch\n'
            f'OCT Retinal Classification (5-class)',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        out = f"{RESULTS}/compare_{arch}_transfer_vs_scratch.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out}")

    # ─────────────────────────────────────────
    # PLOT 3: Final accuracy bar chart
    # Auto-reads from JSON logs — no manual entry
    # ─────────────────────────────────────────
    bar_labels   = []
    bar_accs     = []
    bar_colors   = []
    bar_alphas   = []
    acc_sources  = []

    for key, label in DISPLAY_ORDER:
        if key not in logs:
            continue
        acc, src = get_final_acc(logs[key])
        if acc is None:
            continue
        bar_labels.append(label)
        bar_accs.append(acc * 100)
        bar_colors.append(get_arch_color(key))
        bar_alphas.append(0.4 if 'scratch' in key else 1.0)
        acc_sources.append(src)

    if bar_labels:
        fig, ax = plt.subplots(figsize=(14, 7))

        for i, (label, acc, color, alpha, src) in enumerate(
                zip(bar_labels, bar_accs,
                    bar_colors, bar_alphas, acc_sources)):
            ax.bar(i, acc, color=color, alpha=alpha,
                   edgecolor='white', width=0.6)
            marker = '*' if src == 'val_best' else ''
            ax.text(i, acc + 0.3,
                   f'{acc:.2f}%{marker}',
                   ha='center', va='bottom',
                   fontweight='bold', fontsize=10)

        ax.set_xticks(range(len(bar_labels)))
        ax.set_xticklabels(bar_labels, fontsize=10)

        for y, color, label in REFERENCES:
            ax.axhline(y=y * 100, color=color,
                      linestyle='--', linewidth=2, label=label)

        # Legend
        solid_p = mpatches.Patch(color='gray', alpha=1.0,
                                  label='Transfer Learning')
        faded_p = mpatches.Patch(color='gray', alpha=0.4,
                                  label='From Scratch')
        alex_p  = mpatches.Patch(color=ARCH_COLORS['alexnet'],
                                  label='AlexNet')
        vgg_p   = mpatches.Patch(color=ARCH_COLORS['vgg16'],
                                  label='VGG16')
        res_p   = mpatches.Patch(color=ARCH_COLORS['resnet50'],
                                  label='ResNet50')
        ref_patches = [
            plt.Line2D([0],[0], color=c, linestyle='--', label=l)
            for _, c, l in REFERENCES
        ]
        ax.legend(handles=[solid_p, faded_p,
                           alex_p, vgg_p, res_p] + ref_patches,
                  fontsize=9, ncol=2)

        ax.set_ylim(0, 105)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(
            'Final Test Accuracy — All Models\n'
            'Transfer Learning (solid) vs From Scratch (faded)\n'
            '* = best val accuracy used (no final test accuracy saved)',
            fontsize=13, fontweight='bold'
        )
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        out = f"{RESULTS}/compare_final_accuracy_all.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out}")

    # ─────────────────────────────────────────
    # SUMMARY TABLE
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY — BEST ACCURACY PER MODEL")
    print("=" * 60)
    print(f"{'Model':<35} {'Accuracy':<12} {'Source'}")
    print(f"{'-'*60}")
    for key, log in sorted(logs.items()):
        acc, src = get_final_acc(log)
        if acc:
            label = get_label(key)
            print(f"  {label:<33} {acc:.4f}      {src}")
    print("=" * 60)
    print("\n✓ All plots saved to results/")
    print("  compare_all_training_curves.png")
    print("  compare_alexnet_transfer_vs_scratch.png")
    print("  compare_vgg16_transfer_vs_scratch.png")
    print("  compare_resnet50_transfer_vs_scratch.png")
    print("  compare_final_accuracy_all.png")


if __name__ == '__main__':
    main()