import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# 1. Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 2. Load your dataset
dataset = datasets.ImageFolder("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
dataset = datasets.ImageFolder("C:/Users/heman/Desktop/ai -rural -emergency/backend/dataset", transform=transform)

# 3. Load pre-trained MobileNetV2
model = models.mobilenet_v2(pretrained=True)

# 4. Modify the final layer to match your number of classes
model.classifier[1] = nn.Linear(model.last_channel, len(dataset.classes))

# 5. Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. Training loop (5 epochs for hackathon demo)
for epoch in range(20):
    running_loss = 0.0
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} complete | Loss: {running_loss:.4f}")

import os

# Save the trained model
os.makedirs("ai-models", exist_ok=True)
torch.save(model.state_dict(), "ai-models/images_model.pth")

# Save the class names
with open("ai-models/classes.txt", "w") as f:
    f.write("\n".join(dataset.classes))

print("Model and class names saved successfully.")
