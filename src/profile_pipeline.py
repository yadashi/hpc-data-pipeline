
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from real_dataset import RealImageDataset


def sync_device():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    dataset = RealImageDataset("data/raw/cifar10/images")

    train_dataset, _ = random_split(
        dataset,
        [24, 8],
        generator=torch.Generator().manual_seed(42)
    )

    loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    model.train()

    # Warm-up
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.step()
    sync_device()

    # Profiling measurements
    iterator_startup = 0.0
    data_wait = 0.0
    compute_time = 0.0
    total_images = 0

    for epoch in range(3):

        # Measure iterator startup
        t1 = time.perf_counter()
        iterator = iter(loader)
        iterator_startup += time.perf_counter() - t1

        while True:
            # Measure waiting for next batch
            t1 = time.perf_counter()

            try:
                images, labels = next(iterator)
            except StopIteration:
                break

            data_wait += time.perf_counter() - t1

            # Measure model computation
            t1 = time.perf_counter()

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sync_device()

            compute_time += time.perf_counter() - t1
            total_images += len(images)

    total_time = iterator_startup + data_wait + compute_time

    print("\n===== PIPELINE PROFILING =====")
    print("Device:", device)
    print("Workers:", loader.num_workers)
    print("Images processed:", total_images)

    print("\n--- Time breakdown ---")
    print("Iterator startup:", round(iterator_startup, 4), "sec")
    print("Data waiting:", round(data_wait, 4), "sec")
    print("Model computation:", round(compute_time, 4), "sec")
    print("Total measured time:", round(total_time, 4), "sec")

    print("\n--- Performance ---")
    print("Throughput:", round(total_images / total_time, 2), "images/sec")

    print("\nProfiling completed!")


if __name__ == "__main__":
    main()
