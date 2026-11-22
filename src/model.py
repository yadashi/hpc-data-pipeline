
import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=100):

    model = models.resnet18(weights=None)

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    return model


if __name__ == "__main__":

    # Select device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Selected device:", device)

    # Create model and move it to device
    model = create_model(num_classes=100)
    model = model.to(device)

    # Create a dummy batch of 16 RGB images
    images = torch.randn(16, 3, 224, 224).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(images)

    torch.mps.synchronize() if device.type == "mps" else None

    print("Input shape:", images.shape)
    print("Output shape:", outputs.shape)
    print("Output device:", outputs.device)
    print("Forward pass successful!")
