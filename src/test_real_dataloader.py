
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from real_dataset import RealImageDataset

def main():
    dataset = RealImageDataset("data/raw/cifar10/images")

    train_dataset, val_dataset = random_split(
        dataset, [24, 8]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=8,
        shuffle=True, num_workers=0
    )

    val_loader = DataLoader(
        val_dataset, batch_size=8,
        shuffle=False, num_workers=0
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)

    print("Device:", device)
    print("Input shape:", images.shape)
    print("Model output shape:", outputs.shape)
    print("Number of classes:", model.fc.out_features)
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Forward pass successful!")

if __name__ == "__main__":
    main()
