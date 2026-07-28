# In a completely separate file — inference.py
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# Rebuild the architecture (empty, no weights yet)
model = models.vgg16()
model.classifier[6] = nn.Linear(4096, 5)

# Load your saved weights into it
model.load_state_dict(torch.load('oct_5class_model.pth'))
model.eval()  # inference mode, no training

# Define same transform as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load one new OCT image and get a diagnosis
image = Image.open('new_scan.jpg').convert('RGB')
tensor = transform(image).unsqueeze(0)  # add batch dimension

classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL','NON_OCT']

with torch.no_grad():
    output = model(tensor)
    prediction = output.argmax(1).item()
    confidence = torch.softmax(output, dim=1)[0][prediction].item()

print(f"Diagnosis: {classes[prediction]}")
print(f"Confidence: {confidence:.2%}")