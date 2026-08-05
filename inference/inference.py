# inference.py
# OCT Retinal Disease Classifier — Inference Script
#
# Usage:
#   Single image:    python inference/inference.py path/to/image.jpg
#   Multiple images: python inference/inference.py img1.jpg img2.jpg img3.jpg
#   Folder:          python inference/inference.py path/to/folder/
#   Default test:    python inference/inference.py --test
#   Compare models:  python inference/inference.py --compare path/to/image.jpg
#   Temperature:     python inference/inference.py --temp 2.0 path/to/image.jpg

import os
import sys
import ssl
import certifi
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# ─────────────────────────────────────────
# CONFIGURATION
# Linux:  /home/dantails/PycharmProjects/Conv_Project
# Mac:    /Users/jesusurias/PycharmProjects/Conv_Project
# ─────────────────────────────────────────
BASE         = "/home/dantails/PycharmProjects/Conv_Project"
WEIGHTS_PATH = f"{BASE}/weights/vgg16_5class.pth"
TEST_IMAGES  = f"{BASE}/test_images"
RESULTS_DIR  = f"{BASE}/results/inference"

# Default temperature — tune this value:
# T=1.0 → standard softmax (overconfident)
# T=2.0 → moderate calibration
# T=4.0 → high calibration (more spread)
DEFAULT_TEMP = 2.0

CLASSES_5 = ['CNV', 'DME', 'DRUSEN', 'NORMAL', 'NOT_OCT']
CLASSES_4 = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

DESCRIPTIONS = {
    'CNV':     'Choroidal Neovascularization — abnormal blood vessel '
               'growth beneath the retina. Requires prompt treatment '
               'to prevent vision loss.',
    'DME':     'Diabetic Macular Edema — fluid accumulation in the '
               'macula caused by diabetic retinopathy. A leading '
               'cause of vision impairment in diabetics.',
    'DRUSEN':  'Drusen deposits — small yellow deposits beneath the '
               'retina. An early warning sign of age-related macular '
               'degeneration (AMD). Monitor regularly.',
    'NORMAL':  'Normal retina — no pathology detected. Retinal '
               'layers appear healthy with no signs of disease.',
    'NOT_OCT': 'Not an OCT scan — this image does not appear to be '
               'a retinal OCT scan. Please provide a valid OCT image '
               'for diagnosis.'
}

URGENCY = {
    'CNV':     ('HIGH',    '#d32f2f'),
    'DME':     ('MEDIUM',  '#f57c00'),
    'DRUSEN':  ('LOW',     '#fbc02d'),
    'NORMAL':  ('NONE',    '#2e7d32'),
    'NOT_OCT': ('INVALID', '#757575')
}

COLORS = {
    'CNV':     '#d32f2f',
    'DME':     '#f57c00',
    'DRUSEN':  '#fbc02d',
    'NORMAL':  '#2e7d32',
    'NOT_OCT': '#757575'
}

MODEL_CONFIGS = [
    {
        'name':        'VGG16 4-class (transfer)',
        'weights':     f"{BASE}/weights/vgg16_4class.pth",
        'classes':     CLASSES_4,
        'num_classes': 4,
        'arch':        'vgg16'
    },
    {
        'name':        'VGG16 5-class (transfer)',
        'weights':     f"{BASE}/weights/vgg16_5class.pth",
        'classes':     CLASSES_5,
        'num_classes': 5,
        'arch':        'vgg16'
    },
    {
        'name':        'ResNet50 4-class (transfer)',
        'weights':     f"{BASE}/weights/resnet50_4class.pth",
        'classes':     CLASSES_4,
        'num_classes': 4,
        'arch':        'resnet50'
    },
    {
        'name':        'ResNet50 5-class (transfer)',
        'weights':     f"{BASE}/weights/resnet50_5class.pth",
        'classes':     CLASSES_5,
        'num_classes': 5,
        'arch':        'resnet50'
    },
    {
        'name':        'VGG16 5-class (scratch)',
        'weights':     f"{BASE}/weights/vgg16_5class_scratch.pth",
        'classes':     CLASSES_5,
        'num_classes': 5,
        'arch':        'vgg16'
    },
    {
        'name':        'ResNet50 5-class (scratch)',
        'weights':     f"{BASE}/weights/resnet50_5class_scratch.pth",
        'classes':     CLASSES_5,
        'num_classes': 5,
        'arch':        'resnet50'
    },
]


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def build_model(arch, num_classes, weights_path, device):
    if arch == 'vgg16':
        m = models.vgg16(weights=None)
        m.classifier[6] = nn.Linear(4096, num_classes)
    else:
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(2048, num_classes)
    m.load_state_dict(torch.load(weights_path, map_location=device))
    m.eval()
    return m.to(device)


def apply_temperature(raw_output, temperature=1.0):
    """
    Temperature scaling — divides raw logits before softmax.
    T=1.0 → standard softmax (overconfident for trained models)
    T>1.0 → softer probabilities, more variance expressed
    T<1.0 → even more confident (rarely useful)
    """
    scaled = torch.tensor(raw_output) / temperature
    return torch.softmax(scaled, dim=0).numpy()


def run_prediction(image_path, model, transform, device,
                   classes, temperature=1.0):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return None, None, None, str(e)

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_output = model(tensor)[0].cpu().numpy()

    # Standard softmax (T=1.0)
    probs_standard = apply_temperature(raw_output, temperature=1.0)

    # Temperature scaled
    probs_calibrated = apply_temperature(raw_output, temperature=temperature)

    # Prediction based on calibrated probs
    # (same winner as standard — temperature never changes the ranking)
    pred_idx   = int(probs_calibrated.argmax())
    prediction = classes[pred_idx]

    return prediction, probs_standard.tolist(), \
           probs_calibrated.tolist(), None


def collect_images(args):
    extensions  = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    for arg in args:
        if os.path.isdir(arg):
            for f in sorted(os.listdir(arg)):
                if os.path.splitext(f)[1].lower() in extensions:
                    image_paths.append(os.path.join(arg, f))
        elif os.path.isfile(arg):
            image_paths.append(arg)
        else:
            print(f"WARNING: {arg} not found — skipping")
    return image_paths


# ─────────────────────────────────────────
# TERMINAL OUTPUT
# ─────────────────────────────────────────
def print_result(image_path, prediction, probs_std,
                 probs_cal, classes, temperature):
    filename   = os.path.basename(image_path)
    confidence = probs_cal[classes.index(prediction)]
    urgency_label, _ = URGENCY[prediction]

    print(f"\n{'='*65}")
    print(f"Image:       {filename}")
    print(f"Diagnosis:   {prediction}")
    print(f"Confidence:  {confidence:.4%} "
          f"(temperature={temperature})")
    print(f"Urgency:     {urgency_label}")
    print(f"\nMeaning:")
    print(f"  {DESCRIPTIONS[prediction]}")

    print(f"\n{'─'*65}")
    print(f"{'Class':<12} {'Standard (T=1)':<18} {'Calibrated (T='
          f"{temperature})":<20} {'Raw logit'}")
    print(f"{'─'*65}")

    for i, cls in enumerate(classes):
        std  = probs_std[i]
        cal  = probs_cal[i]
        marker = ' ←' if cls == prediction else ''
        print(f"{cls:<12} {std:<18.4%} {cal:<20.4%}{marker}")

    print(f"\nNote: Temperature scaling (T={temperature}) redistributes")
    print(f"      probability mass without changing the prediction.")
    print(f"      Standard softmax (T=1) often shows false 100% confidence.")
    print(f"{'='*65}")


def print_batch_summary(results):
    print(f"\n{'='*60}")
    print(f"BATCH SUMMARY — {len(results)} images processed")
    print(f"{'='*60}")
    print(f"{'Image':<30} {'Prediction':<12} {'Confidence'}")
    print(f"{'-'*60}")
    for filename, prediction, confidence, error in results:
        if error:
            print(f"{filename:<30} ERROR: {error}")
        else:
            print(f"{filename:<30} {prediction:<12} {confidence:.4%}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────
# VISUAL OUTPUT — single model
# ─────────────────────────────────────────
def save_visual(image_path, prediction, probs_std,
                probs_cal, classes, temperature):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename  = os.path.basename(image_path)
    stem      = os.path.splitext(filename)[0]
    save_path = f"{RESULTS_DIR}/{stem}_result.png"

    confidence       = probs_cal[classes.index(prediction)]
    urgency_label, _ = URGENCY[prediction]

    try:
        original = Image.open(image_path).convert('RGB')
    except:
        return

    # Sort by calibrated probability
    sorted_results = sorted(
        zip(classes, probs_std, probs_cal),
        key=lambda x: x[2]
    )
    display_classes  = [r[0] for r in sorted_results]
    display_std      = [r[1] for r in sorted_results]
    display_cal      = [r[2] for r in sorted_results]
    bar_colors       = [COLORS[c] for c in display_classes]

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor('#1a1a2e')

    # Left — image
    ax_img = fig.add_axes([0.02, 0.1, 0.28, 0.82])
    ax_img.imshow(original)
    ax_img.axis('off')
    ax_img.set_title(f'Input: {filename}',
                     color='white', fontsize=11, pad=8)

    # Middle-left — standard probs
    ax_std = fig.add_axes([0.33, 0.15, 0.22, 0.72])
    ax_std.set_facecolor('#16213e')
    bars = ax_std.barh(display_classes, display_std,
                       color=bar_colors,
                       edgecolor='white', linewidth=0.4,
                       height=0.6)
    for bar, prob in zip(bars, display_std):
        if prob > 0.005:
            ax_std.text(prob + 0.01,
                        bar.get_y() + bar.get_height()/2,
                        f'{prob:.1%}',
                        va='center', ha='left',
                        color='white', fontsize=9)
    ax_std.set_xlim(0, 1.25)
    ax_std.set_title(f'Standard (T=1.0)\nOverconfident',
                     color='#ffaaaa', fontsize=10, pad=6)
    ax_std.set_xlabel('Probability', color='white', fontsize=9)
    ax_std.tick_params(colors='white', labelsize=8)
    for spine in ax_std.spines.values():
        spine.set_color('#333366')
    ax_std.spines['top'].set_visible(False)
    ax_std.spines['right'].set_visible(False)

    # Middle-right — calibrated probs
    ax_cal = fig.add_axes([0.58, 0.15, 0.22, 0.72])
    ax_cal.set_facecolor('#16213e')
    bars = ax_cal.barh(display_classes, display_cal,
                       color=bar_colors,
                       edgecolor='white', linewidth=0.4,
                       height=0.6)
    for bar, prob in zip(bars, display_cal):
        if prob > 0.005:
            ax_cal.text(prob + 0.01,
                        bar.get_y() + bar.get_height()/2,
                        f'{prob:.1%}',
                        va='center', ha='left',
                        color='white', fontsize=9)
    ax_cal.set_xlim(0, 1.25)
    ax_cal.set_title(f'Calibrated (T={temperature})\nMore honest',
                     color='#aaffaa', fontsize=10, pad=6)
    ax_cal.set_xlabel('Probability', color='white', fontsize=9)
    ax_cal.tick_params(colors='white', labelsize=8)
    for spine in ax_cal.spines.values():
        spine.set_color('#333366')
    ax_cal.spines['top'].set_visible(False)
    ax_cal.spines['right'].set_visible(False)

    # Right — diagnosis card
    ax_info = fig.add_axes([0.83, 0.1, 0.15, 0.82])
    ax_info.set_facecolor(COLORS[prediction])
    ax_info.axis('off')

    ax_info.text(0.5, 0.93, 'DIAGNOSIS',
                color='white', fontsize=10,
                ha='center', va='top',
                transform=ax_info.transAxes,
                fontweight='bold')
    ax_info.text(0.5, 0.80, prediction,
                color='white', fontsize=18,
                ha='center', va='top',
                transform=ax_info.transAxes,
                fontweight='bold')
    ax_info.text(0.5, 0.65, f'{confidence:.1%}',
                color='white', fontsize=20,
                ha='center', va='top',
                transform=ax_info.transAxes,
                fontweight='bold')
    ax_info.text(0.5, 0.53, f'T={temperature}',
                color='white', fontsize=9,
                ha='center', va='top',
                transform=ax_info.transAxes, alpha=0.8)
    ax_info.text(0.5, 0.43, 'URGENCY',
                color='white', fontsize=9,
                ha='center', va='top',
                transform=ax_info.transAxes,
                fontweight='bold')
    ax_info.text(0.5, 0.34, urgency_label,
                color='white', fontsize=13,
                ha='center', va='top',
                transform=ax_info.transAxes,
                fontweight='bold')

    desc  = DESCRIPTIONS[prediction]
    words = desc.split()
    lines, current = [], ''
    for word in words:
        if len(current) + len(word) + 1 <= 20:
            current += (' ' if current else '') + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    ax_info.text(0.5, 0.22, '\n'.join(lines[:8]),
                color='white', fontsize=7,
                ha='center', va='top',
                transform=ax_info.transAxes, alpha=0.9)

    fig.text(0.5, 0.97,
             'OCT Retinal Disease Classifier — SJSU AI HUB',
             ha='center', color='white',
             fontsize=12, fontweight='bold')

    plt.savefig(save_path, dpi=150,
                facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print(f"  Visual saved: {save_path}")


# ─────────────────────────────────────────
# MULTI-MODEL COMPARISON MODE
# ─────────────────────────────────────────
def compare_all_models(image_path, device, transform, temperature):
    filename = os.path.basename(image_path)

    print(f"\n{'='*75}")
    print(f"MULTI-MODEL COMPARISON: {filename}  (T={temperature})")
    print(f"{'='*75}")
    print(f"{'Model':<35} {'Pred':<10} {'T=1 (std)':<14} "
          f"{'T={} (cal)'.format(temperature):<14} {'Raw logit'}")
    print(f"{'-'*75}")

    results = []

    for config in MODEL_CONFIGS:
        if not os.path.exists(config['weights']):
            print(f"{config['name']:<35} [weights not found]")
            continue

        try:
            m = build_model(
                config['arch'],
                config['num_classes'],
                config['weights'],
                device
            )
            image  = Image.open(image_path).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                raw_out = m(tensor)[0].cpu().numpy()

            probs_std = apply_temperature(raw_out, 1.0)
            probs_cal = apply_temperature(raw_out, temperature)

            pred_idx   = int(probs_cal.argmax())
            prediction = config['classes'][pred_idx]
            conf_std   = probs_std[pred_idx]
            conf_cal   = probs_cal[pred_idx]
            raw_score  = raw_out[pred_idx]

            results.append({
                'name':       config['name'],
                'prediction': prediction,
                'conf_std':   conf_std,
                'conf_cal':   conf_cal,
                'raw_score':  raw_score,
                'probs_std':  probs_std.tolist(),
                'probs_cal':  probs_cal.tolist(),
                'classes':    config['classes']
            })

            print(f"{config['name']:<35} {prediction:<10} "
                  f"{conf_std:<14.4%} {conf_cal:<14.4%} "
                  f"{raw_score:.4f}")

        except Exception as e:
            print(f"{config['name']:<35} ERROR: {e}")

    # Agreement analysis
    predictions = [r['prediction'] for r in results]
    unique      = set(predictions)

    print(f"\n{'─'*75}")
    if len(unique) == 1:
        print(f"ALL MODELS AGREE: {predictions[0]}")
    else:
        from collections import Counter
        counts = Counter(predictions)
        print(f"MODELS DISAGREE:")
        for pred, count in counts.most_common():
            names = [r['name'] for r in results
                     if r['prediction'] == pred]
            print(f"  {pred}: {count} model(s)")
            for n in names:
                print(f"    → {n}")

    print(f"\nConfidence range (standard): "
          f"{min(r['conf_std'] for r in results):.4%} — "
          f"{max(r['conf_std'] for r in results):.4%}")
    print(f"Confidence range (T={temperature}):   "
          f"{min(r['conf_cal'] for r in results):.4%} — "
          f"{max(r['conf_cal'] for r in results):.4%}")
    print(f"\nKey insight: Standard softmax (T=1) amplifies any")
    print(f"dominant raw score to near-100%. Temperature scaling")
    print(f"(T={temperature}) reveals the true uncertainty beneath.")
    print(f"{'='*75}\n")

    save_comparison_visual(image_path, results, temperature)


def save_comparison_visual(image_path, results, temperature):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename  = os.path.basename(image_path)
    stem      = os.path.splitext(filename)[0]
    save_path = f"{RESULTS_DIR}/{stem}_comparison.png"

    n_models = len(results)
    if n_models == 0:
        return

    try:
        original = Image.open(image_path).convert('RGB')
    except:
        return

    fig = plt.figure(figsize=(18, 3 + n_models * 1.5))
    fig.patch.set_facecolor('#1a1a2e')

    fig.text(0.5, 0.98,
             f'Multi-Model Comparison: {filename}',
             ha='center', color='white',
             fontsize=13, fontweight='bold')
    fig.text(0.5, 0.95,
             f'Left bars: Standard softmax (T=1.0)  |  '
             f'Right bars: Temperature scaled (T={temperature})  |  '
             f'OCT Retinal Disease Classifier — SJSU AI HUB',
             ha='center', color='#aaaacc', fontsize=9)

    # Original image
    ax_img = fig.add_axes([0.01, 0.04, 0.16, 0.88])
    ax_img.imshow(original)
    ax_img.axis('off')
    ax_img.set_title('Input Image', color='white',
                     fontsize=10, pad=6)

    row_height = 0.88 / n_models

    for idx, result in enumerate(results):
        y_pos   = 0.04 + (n_models - 1 - idx) * row_height
        classes = result['classes']
        pred    = result['prediction']
        conf_std = result['conf_std']
        conf_cal = result['conf_cal']

        # Model label
        ax_label = fig.add_axes([0.19, y_pos + 0.005,
                                  0.12, row_height - 0.01])
        ax_label.set_facecolor(COLORS[pred])
        ax_label.axis('off')
        ax_label.text(0.5, 0.70, result['name'],
                     color='white', fontsize=7.5,
                     ha='center', va='center',
                     transform=ax_label.transAxes,
                     fontweight='bold')
        ax_label.text(0.5, 0.38, pred,
                     color='white', fontsize=11,
                     ha='center', va='center',
                     transform=ax_label.transAxes,
                     fontweight='bold')
        ax_label.text(0.5, 0.12,
                     f'T=1: {conf_std:.1%}  →  '
                     f'T={temperature}: {conf_cal:.1%}',
                     color='white', fontsize=7,
                     ha='center', va='center',
                     transform=ax_label.transAxes)

        bar_colors = [COLORS.get(c, '#555555') for c in classes]

        # Standard probs
        ax_std = fig.add_axes([0.33, y_pos + 0.005,
                                0.29, row_height - 0.01])
        ax_std.set_facecolor('#16213e')
        bars_s = ax_std.barh(classes,
                             result['probs_std'],
                             color=bar_colors,
                             edgecolor='#333366',
                             linewidth=0.3, height=0.55)
        for bar, prob in zip(bars_s, result['probs_std']):
            if prob > 0.005:
                ax_std.text(prob + 0.01,
                           bar.get_y() + bar.get_height()/2,
                           f'{prob:.1%}',
                           va='center', ha='left',
                           color='white', fontsize=7)
        ax_std.set_xlim(0, 1.3)
        ax_std.tick_params(colors='white', labelsize=7)
        for spine in ax_std.spines.values():
            spine.set_color('#333366')
        ax_std.spines['top'].set_visible(False)
        ax_std.spines['right'].set_visible(False)
        if idx == n_models - 1:
            ax_std.set_title('Standard (T=1.0) — Overconfident',
                            color='#ffaaaa', fontsize=9, pad=4)

        # Calibrated probs
        ax_cal = fig.add_axes([0.64, y_pos + 0.005,
                                0.34, row_height - 0.01])
        ax_cal.set_facecolor('#16213e')
        bars_c = ax_cal.barh(classes,
                             result['probs_cal'],
                             color=bar_colors,
                             edgecolor='#333366',
                             linewidth=0.3, height=0.55)
        for bar, prob in zip(bars_c, result['probs_cal']):
            if prob > 0.005:
                ax_cal.text(prob + 0.01,
                           bar.get_y() + bar.get_height()/2,
                           f'{prob:.1%}',
                           va='center', ha='left',
                           color='white', fontsize=7)
        ax_cal.set_xlim(0, 1.3)
        ax_cal.tick_params(colors='white', labelsize=7)
        for spine in ax_cal.spines.values():
            spine.set_color('#333366')
        ax_cal.spines['top'].set_visible(False)
        ax_cal.spines['right'].set_visible(False)
        if idx == n_models - 1:
            ax_cal.set_title(
                f'Calibrated (T={temperature}) — More honest',
                color='#aaffaa', fontsize=9, pad=4)

    plt.savefig(save_path, dpi=150,
                facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print(f"  Comparison visual saved: {save_path}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    device    = get_device()
    transform = get_transform()

    # Parse temperature argument
    temperature = DEFAULT_TEMP
    if '--temp' in sys.argv:
        idx = sys.argv.index('--temp')
        try:
            temperature = float(sys.argv[idx + 1])
            sys.argv.pop(idx + 1)
            sys.argv.pop(idx)
        except (IndexError, ValueError):
            print(f"Invalid --temp value. Using default T={DEFAULT_TEMP}")

    print(f"OCT Retinal Disease Classifier")
    print(f"Using device: {device}")
    print(f"Temperature:  {temperature} "
          f"(T=1.0=standard, higher=more calibrated)")

    # ── Compare mode ──────────────────────
    if '--compare' in sys.argv:
        idx = sys.argv.index('--compare')
        if idx + 1 >= len(sys.argv):
            print("Usage: python inference/inference.py "
                  "--compare path/to/image.jpg")
            sys.exit(1)
        image_path = sys.argv[idx + 1]
        if not os.path.exists(image_path):
            print(f"ERROR: {image_path} not found")
            sys.exit(1)
        compare_all_models(image_path, device, transform, temperature)
        sys.exit(0)

    # ── Standard inference ─────────────────
    print(f"Model: VGG16 5-class (transfer learning)")

    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: Weights not found at {WEIGHTS_PATH}")
        print("Run models/vgg16_5class_train.py first.")
        sys.exit(1)

    model = build_model('vgg16', 5, WEIGHTS_PATH, device)

    if len(sys.argv) < 2 or sys.argv[1] == '--test':
        if os.path.exists(TEST_IMAGES):
            image_paths = collect_images([TEST_IMAGES])
            print(f"Running on test_images/ ({len(image_paths)} images)")
        else:
            print("No images specified and test_images/ not found.")
            print("\nUsage:")
            print("  Single:    python inference/inference.py image.jpg")
            print("  Multiple:  python inference/inference.py i1.jpg i2.jpg")
            print("  Folder:    python inference/inference.py folder/")
            print("  Test:      python inference/inference.py --test")
            print("  Compare:   python inference/inference.py "
                  "--compare image.jpg")
            print("  Temp:      python inference/inference.py "
                  "--temp 3.0 image.jpg")
            sys.exit(0)
    else:
        image_paths = collect_images(sys.argv[1:])

    if not image_paths:
        print("No valid images found.")
        sys.exit(0)

    results = []
    for image_path in image_paths:
        prediction, probs_std, probs_cal, error = run_prediction(
            image_path, model, transform, device,
            CLASSES_5, temperature
        )
        filename = os.path.basename(image_path)
        if error:
            print(f"\nERROR processing {filename}: {error}")
            results.append((filename, None, None, error))
        else:
            print_result(image_path, prediction,
                        probs_std, probs_cal, CLASSES_5, temperature)
            save_visual(image_path, prediction,
                       probs_std, probs_cal, CLASSES_5, temperature)
            confidence = probs_cal[CLASSES_5.index(prediction)]
            results.append((filename, prediction, confidence, None))

    if len(image_paths) > 1:
        print_batch_summary(results)


if __name__ == '__main__':
    main()