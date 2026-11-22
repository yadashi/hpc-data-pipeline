
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class RealImageDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = Path(root_dir)

        # Identify all category folders
        self.classes = sorted(
            folder.name
            for folder in self.root_dir.iterdir()
            if folder.is_dir()
        )

        # Map category names to numerical labels
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }

        # Store image paths and corresponding labels
        self.samples = []

        for class_name in self.classes:

            class_dir = self.root_dir / class_name

            for image_path in class_dir.glob("*.jpg"):

                self.samples.append(
                    (image_path, self.class_to_idx[class_name])
                )

        # Convert PIL images into PyTorch tensors
        self.transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        # Read image from SSD
        image = Image.open(image_path).convert("RGB")

        # Convert image to PyTorch tensor
        image = self.transform(image)

        return image, label


if __name__ == "__main__":

    dataset = RealImageDataset(
        "data/raw/cifar10/images"
    )

    print("Total images:", len(dataset))
    print("Classes:", dataset.classes)

    image, label = dataset[0]

    print("Image tensor shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Image size:", image.shape[1:])
    print("Label:", label)
