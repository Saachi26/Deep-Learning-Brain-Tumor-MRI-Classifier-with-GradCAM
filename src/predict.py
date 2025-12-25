import torch
import timm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import os
import random

def predict():
    # 1. SETUP
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Path to the model we saved in the training step
    MODEL_PATH = "models/brain_tumor_efficientnet.pth" 
    TEST_DIR = "./data/Testing"

    CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # 2. LOAD THE MODEL
    print(f"Loading EfficientNet from {MODEL_PATH}...")
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # 3. PICK A RANDOM TEST IMAGE
    while True:
        # Check if directory exists to avoid crashing
        if not os.path.exists(TEST_DIR):
            print(f"Error: {TEST_DIR} does not exist.")
            return

        random_class = random.choice(CLASSES)
        class_path = os.path.join(TEST_DIR, random_class)
        
        # Handle case where folder names might be slightly different (e.g. no_tumor vs notumor)
        if not os.path.exists(class_path): 
            print(f"Skipping {random_class} (Folder not found)... check your folder names!")
            continue
        
        images = os.listdir(class_path)
        if len(images) == 0: continue
        
        image_name = random.choice(images)
        image_path = os.path.join(class_path, image_name)
        break

    print(f"Testing on file: {image_name}")

    # 4. PREPARE IMAGE
    # Must match the Training script (ImageNet Stats)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    raw_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(raw_image).unsqueeze(0).to(DEVICE)

    # 5. RUN PREDICTION
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    predicted_label = CLASSES[pred_idx.item()]
    confidence = conf.item() * 100

    print(f"Prediction: {predicted_label} ({confidence:.2f}%)")
    print(f"Actual: {random_class}")

    # 6. GENERATE GRAD-CAM HEATMAP
    target_layers = [model.conv_head]

    cam = GradCAM(model=model, target_layers=target_layers)
    
    targets = [ClassifierOutputTarget(pred_idx.item())]

    # Run Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Create visualization
    rgb_img = np.array(raw_image.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Left: Original
    ax[0].imshow(rgb_img)
    ax[0].set_title(f"Actual: {random_class}")
    ax[0].axis('off')
    
    # Right: AI Heatmap
    ax[1].imshow(visualization)
    title_color = 'green' if predicted_label == random_class else 'red'
    ax[1].set_title(f"AI: {predicted_label} ({confidence:.1f}%)", color=title_color, fontweight='bold')
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict()