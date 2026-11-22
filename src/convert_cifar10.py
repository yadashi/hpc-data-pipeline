
import pickle
from pathlib import Path
from PIL import Image
import numpy as np

# Input binary file
input_file = Path(
    "data/raw/cifar10/cifar-10-batches-py/data_batch_1"
)

# Output directory
output_dir = Path("data/raw/cifar10/images")

# CIFAR-10 category names
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Read binary data
with open(input_file, "rb") as file:
    batch = pickle.load(file, encoding="bytes")

images = batch[b"data"]
labels = batch[b"labels"]

# Convert first 32 images
for index in range(32):

    image_array = images[index].reshape(3, 32, 32)
    image_array = image_array.transpose(1, 2, 0)

    image = Image.fromarray(image_array)

    label = labels[index]
    class_name = classes[label]

    class_dir = output_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    image_path = class_dir / f"image_{index}.jpg"

    image.save(image_path, quality=95)

print("Conversion completed!")
print("Images converted:", 32)
print("Output directory:", output_dir)
