
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from real_dataset import RealImageDataset


def sync_device():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def benchmark(num_workers):
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
        num_workers=num_workers
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    model.train()

    # Warm-up: exclude from measurements
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.step()
    sync_device()

    data_time = 0.0
    compute_time = 0.0
    total_images = 0

    start = time.perf_counter()

    for epoch in range(3):
        iterator = iter(loader)

        while True:
            t1 = time.perf_counter()

            try:
                images, labels = next(iterator)
            except StopIteration:
                break

            data_time += time.perf_counter() - t1

            t2 = time.perf_counter()

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sync_device()
            compute_time += time.perf_counter() - t2
            total_images += len(images)

    total_time = time.perf_counter() - start

    print(f"\nWorkers: {num_workers}")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Data loading time: {data_time:.3f} seconds")
    print(f"Compute time: {compute_time:.3f} seconds")
    print(f"Throughput: {total_images / total_time:.2f} images/sec")


def main():
    for workers in [0, 1, 2]:
        benchmark(workers)


if __name__ == "__main__":
    main()
