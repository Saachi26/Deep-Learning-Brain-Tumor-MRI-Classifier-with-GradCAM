import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate():
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    MODEL_PATH = "models/brain_tumor_efficientnet_augmented.pth"
    TEST_DIR = "./data/Testing"
    BATCH_SIZE = 32

    # Check if paths exist
    if not os.path.exists(TEST_DIR):
        print(f"❌ Error: Test directory '{TEST_DIR}' not found.")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found.")
        return

    #1.PREPARE DATA
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # ImageFolder loads classes alphabetically, ensuring consistency with training
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Classes found: {test_dataset.classes}")

    # 2. LOAD MODEL
    print(f"Loading model from {MODEL_PATH}...")
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # 3. RUN EVALUATION
    print("Evaluating on Test Set...")
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Save for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. RESULTS
    accuracy = 100 * correct / total
    print(f"\n--------------------------------------------------")
    print(f"✅ FINAL TEST ACCURACY: {accuracy:.2f}%")
    print(f"--------------------------------------------------\n")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

    # 5. PLOT CONFUSION MATRIX
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
    plt.title(f"Confusion Matrix (Acc: {accuracy:.2f}%)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()