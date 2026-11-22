
import time
import torch
from torch.utils.data import DataLoader
from dataset import SyntheticImageDataset


def benchmark(num_workers):

    dataset = SyntheticImageDataset(number_of_images=32)

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=num_workers
    )

    start_time = time.perf_counter()

    total_images = 0

    for images, labels in dataloader:
        total_images += images.size(0)

    end_time = time.perf_counter()

    total_time = end_time - start_time

    throughput = total_images / total_time

    print("\nWorkers:", num_workers)
    print("Total images:", total_images)
    print("Total time:", round(total_time, 4), "seconds")
    print("Images per second:", round(throughput, 2))


if __name__ == "__main__":

    benchmark(num_workers=1)
