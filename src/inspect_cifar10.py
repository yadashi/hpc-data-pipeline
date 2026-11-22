
import pickle

file_path = "data/raw/cifar10/cifar-10-batches-py/data_batch_1"

with open(file_path, "rb") as file:
    batch = pickle.load(file, encoding="bytes")

images = batch[b"data"]
labels = batch[b"labels"]

print("Number of images:", len(images))
print("Image data shape:", images.shape)
print("Number of labels:", len(labels))
print("First 10 labels:", labels[:10])
