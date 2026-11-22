
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from real_dataset import RealImageDataset

def main():
    # 1. Device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # 2. Dataset
    dataset = RealImageDataset("data/raw/cifar10/images")

    train_dataset, val_dataset = random_split(
        dataset,
        [24, 8],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=8,
        shuffle=True, num_workers=0
    )

    val_loader = DataLoader(
        val_dataset, batch_size=8,
        shuffle=False, num_workers=0
    )

    # 3. Model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # 4. Loss and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9
    )

    # 5. Training
    model.train()
    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(
            f"Batch {batch_idx + 1}/{len(train_loader)}, "
            f"Loss: {loss.item():.4f}"
        )

    print("Average training loss:", total_loss / len(train_loader))

    # 6. Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print("Validation accuracy:", correct / total * 100, "%")
    print("Device:", device)
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
