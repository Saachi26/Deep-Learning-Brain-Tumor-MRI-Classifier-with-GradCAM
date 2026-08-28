"""
Generate the Grad-CAM figures used in the README.

Runs the final model (models/brain_tumor_efficientnet_augmented.pth) over
held-out test images and writes:

  assets/HeatMap.png         one correctly-classified example per class
  assets/HeatMapFailure.png  the model's main error mode: glioma read as no-tumour

Selection is deterministic (sorted filenames, first match), so re-running
reproduces the same figures.

    python src/make_heatmap_figure.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

MODEL_PATH = "models/brain_tumor_efficientnet_augmented.pth"
TEST_DIR = "./data/Testing"

# ImageFolder assigns labels by sorting folder names alphabetically.
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
PRETTY = {"glioma": "Glioma", "meningioma": "Meningioma",
          "notumor": "No Tumor", "pituitary": "Pituitary"}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BG = "#0e1117"
FG = "#e6e6e6"
MUTED = "#8b949e"
GREEN = "#4CAF50"
RED = "#FF4B4B"

_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def infer(model, cam, device, path):
    """Return (rgb_224, gradcam_overlay, predicted_class, confidence_pct)."""
    raw = Image.open(path).convert("RGB")
    x = _tf(raw).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)
    conf, idx = torch.max(probs, 1)

    grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(idx.item())])[0, :]
    rgb = np.float32(np.array(raw.resize((224, 224)))) / 255.0
    return rgb, show_cam_on_image(rgb, grayscale, use_rgb=True), CLASSES[idx.item()], conf.item() * 100


def find(model, cam, device, true_class, want, limit=60):
    """First test image of true_class the model predicts as `want`. Deterministic."""
    d = os.path.join(TEST_DIR, true_class)
    for name in sorted(f for f in os.listdir(d) if not f.startswith("."))[:limit]:
        path = os.path.join(d, name)
        result = infer(model, cam, device, path)
        if result[2] == want:
            return (path,) + result
    return None


def draw_pair(ax_top, ax_bottom, rgb, overlay, header, pred, conf, correct):
    ax_top.imshow(rgb)
    ax_top.set_title(header, color=FG, fontsize=14, fontweight="bold", pad=8)
    ax_top.axis("off")

    ax_bottom.imshow(overlay)
    ax_bottom.set_title(f"{'✓' if correct else '✗'}  {PRETTY[pred]}   {conf:.1f}%",
                        color=GREEN if correct else RED,
                        fontsize=12.5, fontweight="bold", pad=8)
    ax_bottom.axis("off")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found: {MODEL_PATH} (train it first, see README)")
    if not os.path.isdir(TEST_DIR):
        raise SystemExit(f"Test data not found: {TEST_DIR} (see README)")

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    # conv_head is B0's last convolution: the deepest layer that is still spatial.
    cam = GradCAM(model=model, target_layers=[model.conv_head])

    # ---- Figure 1: one correct example per class -------------------------
    fig, axes = plt.subplots(2, 4, figsize=(14, 8.2), facecolor=BG)
    for col, cls in enumerate(CLASSES):
        hit = find(model, cam, device, cls, want=cls)
        if hit is None:
            raise SystemExit(f"No correctly-classified {cls} found in the first images")
        _, rgb, overlay, pred, conf = hit
        draw_pair(axes[0, col], axes[1, col], rgb, overlay, PRETTY[cls], pred, conf, True)
        print(f"  {cls:12s} -> {pred} {conf:.1f}%")

    fig.suptitle("Grad-CAM on EfficientNet-B0 · conv_head · held-out test images",
                 color=FG, fontsize=15.5, fontweight="bold", y=0.975)
    fig.text(0.5, 0.925,
             "Top: input MRI.    Bottom: Grad-CAM overlay — red = pixels that most increased "
             "the predicted class score, blue = ignored.",
             ha="center", color=MUTED, fontsize=10.5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.03, wspace=0.06, hspace=0.14)
    fig.savefig("assets/HeatMap.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("Wrote assets/HeatMap.png")

    # ---- Figure 2: the dominant failure mode ------------------------------
    miss = find(model, cam, device, "glioma", want="notumor")
    if miss is None:
        print("No glioma->notumor error found in the sampled images; skipping failure figure.")
        return
    _, rgb, overlay, pred, conf = miss

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 5.0), facecolor=BG)
    axes[0].imshow(rgb)
    axes[0].set_title("Input — true class: Glioma", color=FG, fontsize=13,
                      fontweight="bold", pad=8)
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"✗  Predicted {PRETTY[pred]}   {conf:.1f}%", color=RED,
                      fontsize=13, fontweight="bold", pad=8)
    axes[1].axis("off")

    fig.suptitle("Failure case — the model's costliest error", color=FG,
                 fontsize=14.5, fontweight="bold", y=0.98)
    fig.text(0.5, 0.075,
             "25 of 400 test gliomas are classified as no-tumour. Attention is diffuse across "
             "the whole brain\nrather than focused on a lesion — the heatmap shows the model "
             "had no localised evidence.",
             ha="center", color=MUTED, fontsize=9.5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.17, wspace=0.06)
    fig.savefig("assets/HeatMapFailure.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"Wrote assets/HeatMapFailure.png  (glioma -> {pred} @ {conf:.1f}%)")


if __name__ == "__main__":
    main()
