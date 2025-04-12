import torch
from torchvision import models, transforms
from PIL import Image
import io

# Load class names from a text file
def load_classes():
    with open("ai-models/classes.txt", "r") as f:
        return f.read().splitlines()

# Load the trained PyTorch model
def load_model():
    class_names = load_classes()
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(torch.load("ai-models/image_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Make prediction from image bytes
def predict_image_from_bytes(image_bytes, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]
