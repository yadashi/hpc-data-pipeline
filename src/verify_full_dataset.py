
from pathlib import Path

root = Path("data/processed/cifar10/train")

total_images = 0

print("===== DATASET VERIFICATION =====")

for class_dir in sorted(root.iterdir()):
    if class_dir.is_dir():
        count = len(list(class_dir.glob("*.jpg")))
        total_images += count
        print(f"{class_dir.name}: {count} images")

print("--------------------------------")
print("Total images:", total_images)
print("Total classes:", len([p for p in root.iterdir() if p.is_dir()]))

if total_images == 50000:
    print("Dataset verification successful!")
else:
    print("WARNING: Expected 50000 images.")
