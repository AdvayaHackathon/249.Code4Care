import torch
from torchvision import models, transforms
from PIL import Image
import os

# Load class names
with open("ai-models/classes.txt", "r") as f:
    classes = f.read().splitlines()

# Load model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
model.load_state_dict(torch.load("ai-models/image_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]
