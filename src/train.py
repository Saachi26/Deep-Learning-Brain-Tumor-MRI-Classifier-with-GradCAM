import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm 
import os

def train():
    # 1. SETUP
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001 
    EPOCHS = 5
    NUM_CLASSES = 4

    # 2. DATA
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    data_dir = './data/Training' 
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Directory '{data_dir}' not found. Did you unzip the dataset?")
        return

    full_dataset = datasets.ImageFolder(data_dir, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Data Loaded: {len(train_dataset)} training, {len(val_dataset)} validation.")
    print(f"Classes found: {full_dataset.classes}")

    # 3. MODEL: EFFICIENTNET-B0
    print("Loading EfficientNet-B0...")
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # 4. TRAINING
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 50 == 0:
                print(f"  > Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        train_acc = 100 * correct / total
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
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

        print(f"✅ EPOCH {epoch+1}/{EPOCHS} RESULT:")
        print(f"   Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {avg_val_loss:.4f}")
        print("-" * 60)

    # 5. SAVE
    os.makedirs("models", exist_ok=True)
    save_path = "models/brain_tumor_efficientnet.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    train()