import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm 
import matplotlib.pyplot as plt
import os
import numpy as np

def finetune_augmented():
    # 1. SETUP
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    BATCH_SIZE = 32
    # Lower LR is good for fine-tuning
    LEARNING_RATE = 0.0001
    EPOCHS = 10 

    # 2. TRANSFORMS
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # HEAVY AUGMENTATION
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),    # Left/Right flip 
        transforms.RandomRotation(15),        # Small rotations
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # CLEAN TRANSFORM (For Validation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 3. DATA LOADERS
    data_dir = './data/Training'
    
    train_dataset_full = datasets.ImageFolder(data_dir, transform=train_transform)
    val_dataset_full = datasets.ImageFolder(data_dir, transform=val_transform)

    # Generate indices manually so we can split them consistently
    num_train = len(train_dataset_full)
    indices = list(range(num_train))
    split = int(np.floor(0.8 * num_train))
    
    # Shuffle indices to ensure random split
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:split], indices[split:]

    # Create Subsets: 
    # train_subset pulls from the "Augmented" loader
    # val_subset pulls from the "Clean" loader
    train_subset = Subset(train_dataset_full, train_idx)
    val_subset = Subset(val_dataset_full, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Augmented Training Images: {len(train_subset)}")
    print(f"Clean Validation Images: {len(val_subset)}")

    # 4. LOAD PREVIOUS MODEL
    print("Loading previous model weights...")
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=4)
    
    # Check if previous model exists
    prev_model_path = "models/brain_tumor_efficientnet.pth"
    if os.path.exists(prev_model_path):
        model.load_state_dict(torch.load(prev_model_path, map_location=DEVICE))
        print("✅ Previous weights loaded successfully.")
    else:
        print("⚠️ Warning: Previous model not found. Starting from scratch (ImageNet weights).")
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=4)
        
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # Weight decay helps prevent overfitting on the new augmented data
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("Starting Augmented Fine-Tuning...")
    
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATE ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {avg_val_loss:.4f}")

    # 5. SAVE
    save_path = "models/brain_tumor_efficientnet_augmented.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to '{save_path}'")

    # 6. PLOT
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Acc (Augmented)')
    plt.plot(history['val_acc'], label='Val Acc (Clean)')
    plt.title('Fine-Tuning Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    finetune_augmented()