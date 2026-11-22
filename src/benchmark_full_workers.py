
import time
import torch
from torch.utils.data import DataLoader, Subset
from real_dataset import RealImageDataset


def benchmark(workers):
    dataset = RealImageDataset("data/processed/cifar10/train")
    subset = Subset(dataset, range(320))

    loader = DataLoader(
        subset,
        batch_size=32,
        shuffle=False,
        num_workers=workers
    )

    # Warm-up one batch
    iterator = iter(loader)
    next(iterator)

    total_images = 0

    start = time.perf_counter()

    for images, labels in iterator:
        total_images += len(images)

    # Include the warm-up batch in total image count,
    # but exclude its loading time from the measurement.
    total_images += 32

    elapsed = time.perf_counter() - start

    print(f"\nWorkers: {workers}")
    print(f"Images processed: {total_images}")
    print(f"Measured time: {elapsed:.3f} sec")
    print(f"Throughput: {total_images / elapsed:.2f} images/sec")


def main():
    for workers in [0, 1, 2]:
        benchmark(workers)


if __name__ == "__main__":
    main()
