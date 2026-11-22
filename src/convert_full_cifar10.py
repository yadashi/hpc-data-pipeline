
import pickle
from pathlib import Path
from PIL import Image
import numpy as np

SOURCE = Path("data/raw/cifar10/cifar-10-batches-py")
DESTINATION = Path("data/processed/cifar10/train")

DESTINATION.mkdir(parents=True, exist_ok=True)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

total_images = 0

for batch_number in range(1, 6):
    batch_path = SOURCE / f"data_batch_{batch_number}"

    with open(batch_path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")

    images = batch[b"data"]
    labels = batch[b"labels"]

    for index, (image_data, label) in enumerate(zip(images, labels)):
        class_name = class_names[label]

        class_dir = DESTINATION / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        image = image_data.reshape(3, 32, 32)
        image = image.transpose(1, 2, 0)

        image = Image.fromarray(image.astype(np.uint8))

        filename = f"batch{batch_number}_{index:05d}.jpg"
        image.save(class_dir / filename, quality=95)

        total_images += 1

    print(f"Batch {batch_number} converted: {len(images)} images")

print("\nConversion completed!")
print("Total images:", total_images)
print("Output directory:", DESTINATION)
