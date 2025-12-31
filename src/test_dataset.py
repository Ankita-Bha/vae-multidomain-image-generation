from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

DATA_DIR = "/workspace/data/raw"
BATCH_SIZE = 64

# Common transforms (grayscale)
grayscale_transform = transforms.Compose([
    transforms.ToTensor()
])

# MNIST (Digits)
mnist = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=grayscale_transform
)

# EMNIST (Alphabets - Balanced)
emnist = datasets.EMNIST(
    root=DATA_DIR,
    split="balanced",
    train=True,
    download=True,
    transform=grayscale_transform
)

# Fashion-MNIST (Fashion)
fashion = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=grayscale_transform
)

# DataLoaders (grayscale)
mnist_loader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)
emnist_loader = DataLoader(emnist, batch_size=BATCH_SIZE, shuffle=True)
fashion_loader = DataLoader(fashion, batch_size=BATCH_SIZE, shuffle=True)

# Sanity check (grayscale)
mnist_batch = next(iter(mnist_loader))[0]
emnist_batch = next(iter(emnist_loader))[0]
fashion_batch = next(iter(fashion_loader))[0]

print("MNIST batch shape:", mnist_batch.shape)
print("EMNIST batch shape:", emnist_batch.shape)
print("Fashion-MNIST batch shape:", fashion_batch.shape)

# CelebA (Faces) — RGB, higher resolution
celeba_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

celeba = ImageFolder(
    root=f"{DATA_DIR}/celeba",
    transform=celeba_transform
)

celeba_loader = DataLoader(
    celeba,
    batch_size=BATCH_SIZE,
    shuffle=True
)

celeba_batch = next(iter(celeba_loader))[0]

print("CelebA batch shape:", celeba_batch.shape)
