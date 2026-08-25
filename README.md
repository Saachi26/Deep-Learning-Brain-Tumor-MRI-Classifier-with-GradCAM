# Deep Learning Brain Tumor MRI Classifier

## 🧠 Project Overview

This project is a deep learning system for classifying brain MRI scans into four categories using a fine-tuned EfficientNet-B0 model.  
It achieves **95.19% test accuracy** and **0.995 ROC-AUC** on a held-out, class-balanced test set of 1,600 images, with **100% specificity** (zero healthy scans misflagged as tumors), and includes **Explainable AI (XAI)** using Grad-CAM to visualize model attention.
![UI overview](assets/UIOverview.png)


---

## 🎯 Key Features

- Multi-class classification: Glioma, Meningioma, Pituitary Tumor, No Tumor
- EfficientNet-B0 with transfer learning
- Grad-CAM heatmap visualizations
- Streamlit-based web interface
- Data augmentation for robust training

---

## 📂 Directory Structure

```
BRAINTUMORMRI/
│
├── data/
├── models/
├── src/
│   ├── app.py
│   ├── augmentedFinetune.py
│   ├── evaluate.py
│   ├── finetune.py
│   ├── predict.py
│   ├── report_metrics.py
│   └── train.py
│
├── reports/            # generated metrics, tables, confusion matrices
│   ├── metrics.json
│   └── metrics.md
│
├── assets/
├── venv/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

- Deep Learning: PyTorch, timm
- Model: EfficientNet-B0
- Computer Vision: OpenCV, Torchvision, PIL
- Explainability: pytorch-grad-cam
- Interface: Streamlit
- Data Handling: NumPy, Matplotlib, Scikit-learn

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Saachi26/Deep-Learning-Brain-Tumor-MRI-Classifier-with-GradCAM.git
cd Deep-Learning-Brain-Tumor-MRI-Classifier-with-GradCAM
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure `torch`, `timm`, `streamlit`, `opencv-python`, and `grad-cam` are installed.

---

## 📥 Dataset Setup

The dataset is **not** included in this repo (it is large and license-restricted).
Download the **Brain Tumor MRI Dataset** from Kaggle:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Unzip it into the `data/` folder so the structure looks exactly like this
(folder names are case-sensitive and must match):

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

> The training/evaluation scripts read from `data/Training` and `data/Testing`.

---

## ⚠️ Trained Model Required to Run the App

The Streamlit app and `evaluate.py` load a trained weights file:

```
models/brain_tumor_efficientnet_augmented.pth
```

This file is **not** committed (it is large). You must either:

1. **Train it yourself** (see "Train the Model" below — run the scripts in order), or
2. **Copy your own saved `.pth`** into the `models/` folder with that exact name.

Without this file, `streamlit run src/app.py` will fail to start.

---

## 💻 Usage

### Run the Web Application

```bash
streamlit run src/app.py
```

Upload an MRI image to receive:
- Predicted tumor class
- Grad-CAM heatmap visualization

---

### Train the Model

The scripts form a chain — run them **in this order**:

**Step 1 — Initial training** (downloads ImageNet-pretrained EfficientNet-B0 and
trains on the MRI data; saves `models/brain_tumor_efficientnet.pth`):

```bash
python src/train.py
```

**Step 2 — Augmented fine-tuning** (loads the Step 1 weights, fine-tunes with data
augmentation; saves `models/brain_tumor_efficientnet_augmented.pth` — this is the
file the app uses):

```bash
python src/augmentedFinetune.py
```

> `src/finetune.py` is an earlier, simpler fine-tuning experiment. It is **not**
> part of the final pipeline (the app loads the `_augmented.pth` weights).

---

### Evaluate the Model

```bash
python src/evaluate.py
```

This prints a classification report and shows a confusion matrix for the final model.

For the **full, reproducible metrics report** — every model in `models/` scored on the
held-out test set, written to `reports/metrics.json`, `reports/metrics.md`, and a
confusion matrix per model:

```bash
python src/report_metrics.py
```

![Confusion Matrix](assets/ConfusionMatrix.png)

*Confusion matrix for `brain_tumor_efficientnet_augmented.pth` on the 1,600-image held-out
test set (95.19% accuracy). Regenerate with `python src/report_metrics.py`.*


---

## 📊 Model Performance

All numbers below are **measured on the held-out test set** (1,600 images, 400 per class,
never seen in training or validation) by running:

```bash
python src/report_metrics.py
```

That script writes [`reports/metrics.json`](reports/metrics.json), a full write-up in
[`reports/metrics.md`](reports/metrics.md), and per-model confusion matrices.

### Headline results — `brain_tumor_efficientnet_augmented.pth`

| Metric | Value |
|---|---|
| **Test accuracy** | **95.19%** (1,523 / 1,600) |
| Balanced accuracy | 95.19% |
| Top-2 accuracy | 96.44% |
| Test loss (cross-entropy) | 0.4739 |
| **Macro F1** | **0.9509** |
| Weighted F1 | 0.9509 |
| Macro precision / recall | 0.9551 / 0.9519 |
| **ROC-AUC** (one-vs-rest, macro) | **0.9950** |
| Cohen's κ | 0.9358 |
| Matthews correlation coef. | 0.9374 |
| Inference latency | 14.96 ms/image (66.9 img/s, Apple Silicon MPS) |
| Model size | 4.01M params / 16.35 MB |

### Per-class results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| glioma | 0.9970 | 0.8300 | 0.9059 | 400 |
| meningioma | 0.8991 | 0.9800 | 0.9378 | 400 |
| notumor | 0.9390 | **1.0000** | 0.9685 | 400 |
| pituitary | 0.9852 | 0.9975 | 0.9913 | 400 |

Glioma is the hardest class (83.0% recall — most confusions land on meningioma); every
other class scores ≥98% recall.

### Tumor vs. no-tumor (binary view)

| Metric | Value |
|---|---|
| Sensitivity | 97.83% |
| **Specificity** | **100.00%** |
| False positives | **0** |
| False negatives | 26 / 1,200 |

### Ablation — effect of augmented fine-tuning

| Metric | Stage 1 (`train.py`) | Stage 2 (`augmentedFinetune.py`) | Δ |
|---|---|---|---|
| Test accuracy | 93.06% | **95.19%** | **+2.13 pts** |
| Macro F1 | 0.9296 | **0.9509** | +0.0213 |
| Misclassified | 111 | **77** | **−31% errors** |
| glioma recall | 0.7675 | **0.8300** | +0.0625 |
| False positives | 3 | **0** | −3 |

### Dataset

| Split | glioma | meningioma | notumor | pituitary | Total |
|---|---|---|---|---|---|
| Training pool | 1,400 | 1,400 | 1,400 | 1,400 | 5,600 |
| Test (held out) | 400 | 400 | 400 | 400 | 1,600 |
| | | | | | **7,200** |

The training pool is split 80/20 at train time → 4,480 train / 1,120 validation.

### Training configuration

| Item | Value |
|---|---|
| Architecture | EfficientNet-B0 (`timm`), ImageNet-pretrained |
| Input | 224 × 224 RGB, ImageNet normalization |
| Loss / Optimizer | Cross-entropy / AdamW |
| Stage 1 | LR 1e-3, 5 epochs, batch 32 |
| Stage 2 | LR 1e-4, weight decay 1e-4, 10 epochs, batch 32 |
| Stage 2 augmentation | HFlip, Rotation(15°), Affine translate 0.1, ColorJitter(0.2, 0.2) |

> **Note on the earlier 99.39% figure.** That number was measured on the *original*
> 7,023-image version of the Kaggle dataset, which its author has since replaced. It is
> not reproducible against the data this repo now uses. **95.19% is the reproducible
> result** on the current 7,200-image version, and is the number this project reports.

---

## 🔍 Explainability with Grad-CAM

Grad-CAM heatmaps are overlaid on MRI images to visualize model attention:

- Red regions: High attention (likely tumor regions)
- Blue regions: Low attention (background or healthy tissue)
![Heatmap Visualization](assets/HeatMap.png)


---

## 🤝 Contributing

Contributions are welcome.  
Please open an issue or submit a pull request for improvements or bug fixes.

---

## 📜 License

This project is licensed under the MIT License.
