
from torch.utils.data import DataLoader, random_split
from real_dataset import RealImageDataset

def main():
    dataset = RealImageDataset("data/processed/cifar10/train")

    train_size = 45000
    val_size = 5000

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    print("Total images:", len(dataset))
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    images, labels = next(iter(train_loader))

    print("Batch shape:", images.shape)
    print("Label shape:", labels.shape)

if __name__ == "__main__":
    main()
