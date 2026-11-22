
import torch
import torchvision
import platform

print("Python platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)

print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Selected device:", device)

# Test GPU computation
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

z = x @ y

print("Matrix multiplication successful!")
print("Result device:", z.device)
print("Result shape:", z.shape)
