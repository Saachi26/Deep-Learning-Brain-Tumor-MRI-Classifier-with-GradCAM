# Deep Learning Brain Tumor MRI Classifier

## 🧠 Project Overview

This project is a deep learning system for classifying brain MRI scans into four categories using a fine-tuned EfficientNet-B0 model.  
It achieves **99.39% test accuracy** and includes **Explainable AI (XAI)** using Grad-CAM to visualize model attention.

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
│   └── train.py
│
├── venv/
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
git clone https://github.com/YOUR_USERNAME/Explainable-Deep-Learning-Brain-Tumor-MRI-Classifier.git
cd Explainable-Deep-Learning-Brain-Tumor-MRI-Classifier
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

Ensure `torch`, `timm`, `streamlit`, `opencv-python`, and `pytorch-grad-cam` are installed.

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

With data augmentation:

```bash
python src/augmentedFinetune.py
```

Standard training:

```bash
python src/train.py
```

---

### Evaluate the Model

```bash
python src/evaluate.py
```

This generates classification reports and confusion matrices.

---

## 📊 Model Performance

- Architecture: EfficientNet-B0
- Weights File: brain_tumor_efficientnet_augmented.pth
- Test Accuracy: **99.39%**

---

## 🔍 Explainability with Grad-CAM

Grad-CAM heatmaps are overlaid on MRI images to visualize model attention:

- Red regions: High attention (likely tumor regions)
- Blue regions: Low attention (background or healthy tissue)

---

## 🤝 Contributing

Contributions are welcome.  
Please open an issue or submit a pull request for improvements or bug fixes.

---

## 📜 License

This project is licensed under the MIT License.
