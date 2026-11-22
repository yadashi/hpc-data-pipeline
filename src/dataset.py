
import torch
from torch.utils.data import Dataset


class SyntheticImageDataset(Dataset):

    def __init__(self, number_of_images=100, num_classes=100):
        self.images = torch.randn(number_of_images, 3, 224, 224)
        self.labels = torch.randint(
            0, num_classes, (number_of_images,)
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        return image, label


if __name__ == "__main__":
    dataset = SyntheticImageDataset()

    print("Total images:", len(dataset))

    image, label = dataset[0]

    print("Image shape:", image.shape)
    print("Label:", label)
