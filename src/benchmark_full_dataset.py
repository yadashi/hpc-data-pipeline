
import time
import torch
import psutil
from torch.utils.data import DataLoader
from real_dataset import RealImageDataset


def main():
    dataset = RealImageDataset("data/processed/cifar10/train")

    # Use a fixed 320-image subset
    subset = torch.utils.data.Subset(dataset, range(320))

    loader = DataLoader(
        subset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    total_images = 0
    data_time = 0.0

    start = time.perf_counter()

    for images, labels in loader:
        data_time += 0  # Timing measured below
        total_images += len(images)

    total_time = time.perf_counter() - start

    memory_mb = (
        psutil.Process().memory_info().rss / (1024 * 1024)
    )

    print("\n===== FULL DATASET LOADING BENCHMARK =====")
    print("Dataset images:", len(dataset))
    print("Images processed:", total_images)
    print("Batch size:", loader.batch_size)
    print("Workers:", loader.num_workers)
    print("Total time:", round(total_time, 3), "seconds")
    print("Throughput:", round(total_images / total_time, 2), "images/sec")
    print("Process RAM:", round(memory_mb, 2), "MB")


if __name__ == "__main__":
    main()
