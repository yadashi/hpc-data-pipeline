
from torch.utils.data import DataLoader
from dataset import SyntheticImageDataset


def main():
    # Create dataset
    dataset = SyntheticImageDataset(number_of_images=32)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=1
    )

    print("Total batches:", len(dataloader))

    # Fetch one batch
    images, labels = next(iter(dataloader))

    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)


if __name__ == "__main__":
    main()
