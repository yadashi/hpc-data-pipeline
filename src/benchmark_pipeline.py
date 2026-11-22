
import time
import psutil
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
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001
    )

    # Warm-up
    model.train()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.step()
    sync_device()

    # Benchmark
    total_data_time = 0.0
    total_compute_time = 0.0
    total_images = 0
    repeats = 3

    start_total = time.perf_counter()

    for epoch in range(repeats):
        iterator = iter(loader)

        while True:
            data_start = time.perf_counter()

            try:
                images, labels = next(iterator)
            except StopIteration:
                break

            total_data_time += time.perf_counter() - data_start

            compute_start = time.perf_counter()

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sync_device()
            total_compute_time += time.perf_counter() - compute_start

            total_images += len(images)

    total_time = time.perf_counter() - start_total
    throughput = total_images / total_time

    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)

    print("\n===== PIPELINE BENCHMARK =====")
    print("Device:", device)
    print("Workers:", loader.num_workers)
    print("Images processed:", total_images)
    print("Total benchmark time:", round(total_time, 3), "seconds")
    print("Data loading time:", round(total_data_time, 3), "seconds")
    print("Model compute time:", round(total_compute_time, 3), "seconds")
    print("Throughput:", round(throughput, 2), "images/sec")
    print("Process RAM:", round(memory_mb, 2), "MB")


if __name__ == "__main__":
    main()
